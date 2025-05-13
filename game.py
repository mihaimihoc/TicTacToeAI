import tkinter as tk
import numpy as np
from tkinter import font, messagebox
import random
import os
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


MODEL_FILE = 'tic_tac_toe_ai.weights.h5'

class TicTacToeAI:
    def __init__(self, input_size=9, output_size=9):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = 0.0   
        self.model = self._build_model()
        self.load_model()

    def _build_model(self):
        m = Sequential()
        m.add(Input(shape=(self.input_size,)))
        m.add(Dense(64, activation='relu'))
        m.add(Dense(64, activation='relu'))
        m.add(Dense(self.output_size, activation='linear'))
        m.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return m

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            self.model.load_weights(MODEL_FILE)

    def act(self, state, available_moves):
        state = np.reshape(state, [1, self.input_size])
        q = self.model.predict(state, verbose=0)[0]
        mask = np.full(self.output_size, -np.inf)
        mask[available_moves] = q[available_moves]
        return int(np.argmax(mask))

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        self.root.geometry("700x700")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        
        self.user_score = 0
        self.ai_score = 0
        self.user_symbol = ''
        self.ai_symbol = ''
        self.current_player = ''
        self.board = [''] * 9
        self.game_active = False
        self.message_timeout_id = None
        
        self.bg_color = '#f0f0f0'
        self.board_color = '#ffffff'
        self.line_color = '#333333'
        self.x_color = '#e74c3c'
        self.o_color = '#3498db'
        self.text_color = '#2c3e50'
        self.highlight_color = '#f1c40f'
        self.button_color = '#2c3e50'
        self.button_text_color = '#ffffff'
        
        self.ai_agent = TicTacToeAI()
        self.title_font = font.Font(family='Helvetica', size=36, weight='bold')
        self.score_font = font.Font(family='Helvetica', size=24)
        self.symbol_font = font.Font(family='Helvetica', size=28, weight='bold')
        self.message_font = font.Font(family='Helvetica', size=20)
        self.cell_font = font.Font(family='Helvetica', size=80, weight='bold')
        self.button_font = font.Font(family='Helvetica', size=20, weight='bold')
        
        self.create_main_menu()
    
    def create_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        if self.message_timeout_id:
            self.root.after_cancel(self.message_timeout_id)
            self.message_timeout_id = None
        
        container = tk.Frame(self.root, bg=self.bg_color)
        container.place(relx=0.5, rely=0.5, anchor='center')
        
        title_label = tk.Label(
            container, 
            text="Tic Tac Toe", 
            font=self.title_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        title_label.pack(pady=(0, 60))
        
        play_button = tk.Button(
            container, 
            text="Play", 
            font=self.button_font, 
            width=10,
            bg=self.button_color, 
            fg=self.button_text_color,
            command=self.create_symbol_selection_screen
        )
        play_button.pack(pady=20)
        
        exit_button = tk.Button(
            container, 
            text="Exit", 
            font=self.button_font, 
            width=10,
            bg=self.button_color, 
            fg=self.button_text_color,
            command=self.root.destroy
        )
        exit_button.pack(pady=20)
    
    def create_symbol_selection_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        if self.message_timeout_id:
            self.root.after_cancel(self.message_timeout_id)
            self.message_timeout_id = None
        
        container = tk.Frame(self.root, bg=self.bg_color)
        container.place(relx=0.5, rely=0.5, anchor='center')
        
        title_label = tk.Label(
            container, 
            text="Tic Tac Toe", 
            font=self.title_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        title_label.pack(pady=(0, 40))
        
        question_label = tk.Label(
            container, 
            text="Choose your symbol:", 
            font=self.message_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        question_label.pack(pady=(0, 20))
        
        button_frame = tk.Frame(container, bg=self.bg_color)
        button_frame.pack()
        
        x_button = tk.Button(
            button_frame, 
            text="X", 
            font=self.symbol_font, 
            width=4, 
            height=1,
            bg=self.x_color, 
            fg='white',
            command=lambda: self.set_symbol('X')
        )
        x_button.pack(side='left', padx=20)
        
        o_button = tk.Button(
            button_frame, 
            text="O", 
            font=self.symbol_font, 
            width=4, 
            height=1,
            bg=self.o_color, 
            fg='white',
            command=lambda: self.set_symbol('O')
        )
        o_button.pack(side='left', padx=20)
        
        back_button = tk.Button(
            container, 
            text="Back", 
            font=self.button_font, 
            width=10,
            bg=self.button_color, 
            fg=self.button_text_color,
            command=self.create_main_menu
        )
        back_button.pack(pady=(40, 0))
    
    def set_symbol(self, symbol):
        self.user_symbol = symbol
        self.ai_symbol = 'O' if symbol == 'X' else 'X'
        self.create_game_screen()
    
    def create_game_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        if self.message_timeout_id:
            self.root.after_cancel(self.message_timeout_id)
            self.message_timeout_id = None
        
        score_frame = tk.Frame(self.root, bg=self.bg_color)
        score_frame.pack(fill='x', padx=20, pady=20)
        
        ai_score_label = tk.Label(
            score_frame, 
            text=f"AI: {self.ai_score}", 
            font=self.score_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        ai_score_label.pack(side='left')
        
        user_score_label = tk.Label(
            score_frame, 
            text=f"Player: {self.user_score}", 
            font=self.score_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        user_score_label.pack(side='right')
        
        self.message_label = tk.Label(
            self.root, 
            text="", 
            font=self.message_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        self.message_label.pack(pady=(10, 30))
        
        self.board_frame = tk.Frame(self.root, bg=self.bg_color)
        self.board_frame.pack(expand=True)
        
        self.cells = []
        for i in range(9):
            row, col = divmod(i, 3)
            cell = tk.Button(
                self.board_frame, 
                text="", 
                font=self.cell_font, 
                width=3, 
                height=1,
                bg=self.board_color, 
                fg=self.text_color,
                relief='ridge',
                borderwidth=2,
                command=lambda idx=i: self.handle_click(idx)
            )
            cell.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            self.cells.append(cell)
        
        for i in range(3):
            self.board_frame.grid_rowconfigure(i, weight=1)
            self.board_frame.grid_columnconfigure(i, weight=1)
        
        self.start_new_game()
    
    def start_new_game(self):
        self.board = [''] * 9
        for cell in self.cells:
            cell.config(text="", bg=self.board_color)
        
        if random.choice([True, False]):
            self.current_player = 'user'
            self.show_message("Your turn!", self.user_symbol)
        else:
            self.current_player = 'ai'
            self.show_message("AI is thinking...", self.ai_symbol)
            self.root.after(1000, self.ai_move)
        
        self.game_active = True
    
    def show_message(self, text, symbol=None):
        if symbol == 'X':
            color = self.x_color
        elif symbol == 'O':
            color = self.o_color
        else:
            color = self.text_color
        
        self.message_label.config(text=text, fg=color)
        
        if self.message_timeout_id:
            self.root.after_cancel(self.message_timeout_id)
        
        if "turn" in text or "thinking" in text:
            self.message_timeout_id = self.root.after(3000, self.clear_message)
    
    def clear_message(self):
        if hasattr(self, 'message_label') and self.message_label.winfo_exists():
            self.message_label.config(text="")
        self.message_timeout_id = None
    
    def handle_click(self, index):
        if not self.game_active or self.current_player != 'user' or self.board[index] != '':
            return
        
        self.make_move(index, self.user_symbol)
        
        if self.check_winner(self.user_symbol):
            self.user_score += 1
            self.game_active = False
            self.show_message("You win!", self.user_symbol)
            self.highlight_winning_cells(self.user_symbol)
            self.root.after(2000, self.show_game_over_screen)
            return
        
        if self.check_tie():
            self.game_active = False
            self.show_message("It's a tie!")
            self.root.after(2000, self.show_game_over_screen)
            return
        
        self.current_player = 'ai'
        self.show_message("AI is thinking...", self.ai_symbol)
        self.root.after(1000, self.ai_move)
    
    def ai_move(self):
        if not self.game_active or self.current_player != 'ai':
            return
        
        state = np.array([ 
            1 if c == self.ai_symbol 
            else -1 if c == self.user_symbol 
            else 0 
            for c in self.board
        ])
        
        available_moves = [i for i, c in enumerate(self.board) if c == '']
        if available_moves:
            move = self.ai_agent.act(state, available_moves)
            self.make_move(move, self.ai_symbol)
            
            if self.check_winner(self.ai_symbol):
                self.ai_score += 1
                self.game_active = False
                self.show_message("AI wins!", self.ai_symbol)
                self.highlight_winning_cells(self.ai_symbol)
                self.root.after(2000, self.show_game_over_screen)
                return
            
            if self.check_tie():
                self.game_active = False
                self.show_message("It's a tie!")
                self.root.after(2000, self.show_game_over_screen)
                return
            
            self.current_player = 'user'
            self.show_message("Your turn!", self.user_symbol)
    
    def make_move(self, index, symbol):
        self.board[index] = symbol
        self.cells[index].config(text=symbol)
        self.cells[index].config(fg=self.x_color if symbol == 'X' else self.o_color)
    
    def check_winner(self, symbol):
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] == symbol:
                return True
        
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] == symbol:
                return True
        
        if self.board[0] == self.board[4] == self.board[8] == symbol:
            return True
        if self.board[2] == self.board[4] == self.board[6] == symbol:
            return True
        
        return False
    
    def highlight_winning_cells(self, symbol):
        winning_cells = []
        
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] == symbol:
                winning_cells.extend([i, i+1, i+2])
        
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] == symbol:
                winning_cells.extend([i, i+3, i+6])
        
        if self.board[0] == self.board[4] == self.board[8] == symbol:
            winning_cells.extend([0, 4, 8])
        if self.board[2] == self.board[4] == self.board[6] == symbol:
            winning_cells.extend([2, 4, 6])
        
        for cell in winning_cells:
            self.cells[cell].config(bg=self.highlight_color)
    
    def check_tie(self):
        return '' not in self.board and not self.check_winner(self.user_symbol) and not self.check_winner(self.ai_symbol)
    
    def show_game_over_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        if self.message_timeout_id:
            self.root.after_cancel(self.message_timeout_id)
            self.message_timeout_id = None
        
        container = tk.Frame(self.root, bg=self.bg_color)
        container.place(relx=0.5, rely=0.5, anchor='center')
        
        title_label = tk.Label(
            container, 
            text="Tic Tac Toe", 
            font=self.title_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        title_label.pack(pady=(0, 60))
        
        score_frame = tk.Frame(container, bg=self.bg_color)
        score_frame.pack(pady=(0, 40))
        
        ai_score_label = tk.Label(
            score_frame, 
            text=f"AI: {self.ai_score}", 
            font=self.score_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        ai_score_label.pack(side='left', padx=20)
        
        user_score_label = tk.Label(
            score_frame, 
            text=f"Player: {self.user_score}", 
            font=self.score_font, 
            bg=self.bg_color, 
            fg=self.text_color
        )
        user_score_label.pack(side='right', padx=20)
        
        play_button = tk.Button(
            container, 
            text="Play Again", 
            font=self.button_font, 
            width=15,
            bg=self.button_color, 
            fg=self.button_text_color,
            command=self.create_symbol_selection_screen
        )
        play_button.pack(pady=20)
        
        exit_button = tk.Button(
            container, 
            text="Exit", 
            font=self.button_font, 
            width=15,
            bg=self.button_color, 
            fg=self.button_text_color,
            command=self.root.destroy
        )
        exit_button.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()