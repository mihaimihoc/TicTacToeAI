# 🧠 Tic Tac Toe AI — Human vs. Self-Trained Neural Network

This project features an advanced Tic Tac Toe game where you play against an AI trained using deep Q-learning with neural networks.

## 📂 Project Structure

### `game.py`
A polished, user-friendly GUI built with **Tkinter**. Features:
- Choose your side (❌ or ⭕)
- Score tracking and game status display
- Smooth transitions and animations
- Centered 900x900 window layout
- The AI never plays randomly—it plays smart and learns from experience

### `model.py`
A complete deep reinforcement learning pipeline using **TensorFlow/Keras**:
- The AI trains via self-play against a rule-based opponent
- Uses Deep Q-Networks (DQN) with experience replay
- Evaluates thousands of valid board states
- Saves trained weights to `tic_tac_toe_ai.weights.h5` for reuse

---

## 🧠 How It Works

- The AI explores the game space by simulating tens of thousands of games
- Experience replay ensures past decisions influence future moves
- A smart ε-greedy strategy balances exploration vs. exploitation

---

## 📁 Files Included

- `game.py` — GUI to play against the AI
- `model.py` — AI training script
- `tic_tac_toe_ai.weights.h5` — Pre-trained model weights
- `experience.pkl` (auto-generated) — Saved experience memory for faster learning

---

## 🧩 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tic-tac-toe-ai.git
cd tic-tac-toe-ai
pip install -r requirements.txt
