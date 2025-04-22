import tkinter as tk
from tkinter import messagebox
import random

# --- Config ---
root = tk.Tk()
root.title("🎮 Tic Tac Toe - You vs Computer")
root.resizable(False, False)
root.configure(bg="#ffe6f0")

COLORS = {
    "font": ("Comic Sans MS", 22, "bold"),
    "x": "#0077b6",  # Player
    "o": "#d62828",  # Computer
    "button_bg": "#fff8dc",
    "active": "#fddde6",
    "title": "#ff6f61",
    "restart": "#8ecae6",
}

board = [["" for _ in range(3)] for _ in range(3)]
buttons = [[None]*3 for _ in range(3)]

def check_win(p):
    return any(
        all(board[i][j] == p for j in range(3)) or
        all(board[j][i] == p for j in range(3)) for i in range(3)
    ) or all(board[i][i] == p for i in range(3)) or all(board[i][2 - i] == p for i in range(3))

def check_draw():
    return all(cell for row in board for cell in row)

def end_game(msg):
    messagebox.showinfo("Game Over", msg)
    for row in buttons:
        for btn in row:
            btn.config(state="disabled")

def computer_move():
    # Try to win or block
    for player in ["O", "X"]:
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = player
                    if check_win(player):
                        board[i][j] = "O"
                        buttons[i][j].config(text="O", fg=COLORS["o"], state="disabled")
                        return
                    board[i][j] = ""
    # Else random move
    empty = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ""]
    if empty:
        i, j = random.choice(empty)
        board[i][j] = "O"
        buttons[i][j].config(text="O", fg=COLORS["o"], state="disabled")

def on_click(i, j):
    if board[i][j] == "":
        board[i][j] = "X"
        buttons[i][j].config(text="X", fg=COLORS["x"], state="disabled")
        if check_win("X"):
            end_game("🎉 You win!")
        elif check_draw():
            end_game("It's a draw!")
        else:
            root.after(300, computer_turn)

def computer_turn():
    computer_move()
    if check_win("O"):
        end_game("💻 Computer wins!")
    elif check_draw():
        end_game("It's a draw!")

def restart():
    for i in range(3):
        for j in range(3):
            board[i][j] = ""
            buttons[i][j].config(text="", state="normal", bg=COLORS["button_bg"],
                                 activebackground=COLORS["active"])

# --- UI Setup ---
tk.Label(root, text="Tic Tac Toe", font=("Comic Sans MS", 26, "bold"),
         fg=COLORS["title"], bg="#ffe6f0").grid(row=0, column=0, columnspan=3, pady=15)

for i in range(3):
    for j in range(3):
        btn = tk.Button(root, font=COLORS["font"], width=5, height=2,
                        bg=COLORS["button_bg"], activebackground=COLORS["active"],
                        command=lambda i=i, j=j: on_click(i, j))
        btn.grid(row=i+1, column=j, padx=8, pady=8)
        buttons[i][j] = btn

tk.Button(root, text="🔁 Restart", font=("Arial", 14, "bold"),
          bg=COLORS["restart"], fg="white", activebackground="#219ebc",
          command=restart).grid(row=4, column=0, columnspan=3, pady=12)

root.mainloop()
