import numpy as np
import tensorflow as tf
import random
import os
import pickle
import argparse
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)

tf.config.list_physical_devices('GPU')
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

MODEL_FILE = 'tic_tac_toe_ai.weights.h5'
MEMORY_FILE = 'experience.pkl'

class TicTacToeAI:
    def __init__(self, input_size=9, output_size=9):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = 0.95
        self.epsilon = 0.1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.001
        self.memory = deque(maxlen=20000)
        self.model = self._build_model()
        if os.path.exists(MEMORY_FILE):
            self.load_memory()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, available_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_moves)
        state = np.reshape(state, [1, self.input_size])
        act_values = self.model.predict(state, verbose=0)[0]
        masked = np.full(self.output_size, -float('inf'))
        masked[available_moves] = act_values[available_moves]
        return int(np.argmax(masked))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([s for (s, a, r, ns, d) in minibatch])
        targets = self.model.predict(states, verbose=0)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done or next_state is None:
                targets[idx][action] = reward
            else:
                next_state = np.reshape(next_state, [1, self.input_size])
                future_q = np.amax(self.model.predict(next_state, verbose=0)[0])
                targets[idx][action] = reward + self.gamma * future_q
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save_weights(MODEL_FILE)
        with open(MEMORY_FILE, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.model.load_weights(MODEL_FILE)

    def load_memory(self):
        try:
            with open(MEMORY_FILE, 'rb') as f:
                mem = pickle.load(f)
                self.memory = deque(mem, maxlen=self.memory.maxlen)
        except Exception:
            print("Error loading memory, starting fresh.")

class Trainer:
    def __init__(self, episodes=50000):
        self.ai = TicTacToeAI()
        self.user_symbol = 'O'
        self.ai_symbol = 'X'
        self.episodes = episodes

    def train(self):
        print(Fore.CYAN + f"Starting training for {self.episodes} episodes...")
        wins, losses, ties = 0, 0, 0
        try:
            
            for episode in tqdm(range(1, self.episodes + 1), desc="Training", unit="game"):
                board = [''] * 9
                current = random.choice(['ai', 'user'])
                done = False
                while not done:
                    state = np.array([1 if c == self.ai_symbol else -1 if c == self.user_symbol else 0 for c in board])
                    available = [i for i, c in enumerate(board) if c == '']
                    if not available:
                        break
                    if current == 'ai':
                        action = self.ai.act(state, available)
                        board[action] = self.ai_symbol
                        reward, done = self._evaluate(board, self.ai_symbol)
                        next_state = None if done else np.array([1 if c == self.ai_symbol else -1 if c == self.user_symbol else 0 for c in board])
                        self.ai.remember(state, action, reward, next_state, done)
                        if done:
                            if reward == 10:
                                print(Fore.GREEN + f"Episode {episode}: AI won")
                                wins += 1
                            elif reward == 0:
                                print(Fore.YELLOW + f"Episode {episode}: Tie")
                                ties += 1
                            elif reward == -10:
                                print(Fore.RED + f"Episode {episode}: AI lost")
                                losses += 1
                            break
                        current = 'user'
                    else:
                        action = smart_opponent_move(board.copy(), self.user_symbol)
                        board[action] = self.user_symbol
                        reward, done = self._evaluate(board, self.user_symbol)
                        if done:
                            self.ai.remember(state, action, -10, None, True)
                            print(Fore.RED + f"Episode {episode}: AI lost")
                            losses += 1
                            break
                        current = 'ai'
                self.ai.replay()
                if episode % 50 == 0:
                        self.ai.save_model()
                if episode % 1000 == 0:
                    print(Fore.GREEN + f"Ep {episode}: Wins={wins}, Losses={losses}, Ties={ties}, ε={self.ai.epsilon:.4f}")
            self.ai.save_model()
        except KeyboardInterrupt:
            print(Fore.CYAN + "\nTraining interrupted by user! Saving progress...")
        finally:
            self.ai.save_model()
            print(Fore.GREEN + "Progress saved successfully!")

    def _evaluate(self, board, symbol):
        if check_win(board, symbol):
            return (10, True) if symbol == self.ai_symbol else (-10, True)
        if '' not in board:
            return (0, True)
        return (0, False)


def smart_opponent_move(board, symbol):
    for i in range(9):
        if board[i] == '':
            board[i] = symbol
            if check_win(board, symbol): return i
            board[i] = ''
    opp = 'X' if symbol == 'O' else 'O'
    for i in range(9):
        if board[i] == '':
            board[i] = opp
            if check_win(board, opp): board[i] = ''; return i
            board[i] = ''
    if board[4] == '': return 4
    corners = [i for i in [0,2,6,8] if board[i]=='']
    if corners: return random.choice(corners)
    return random.choice([i for i,c in enumerate(board) if c==''])


def check_win(board, symbol):
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    return any(all(board[i]==symbol for i in combo) for combo in wins)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tic Tac Toe AI")
    parser.add_argument('--episodes', type=int, default=50000, help='Number of training games')
    args = parser.parse_args()
    trainer = Trainer(episodes=args.episodes)
    trainer.train()
