import streamlit as st
import heapq
import time

# ----------- 8-Puzzle Solver -----------
SIZE = 3

def manhattan(board, goal):
    distance = 0
    for i in range(SIZE):
        for j in range(SIZE):
            val = board[i][j]
            if val != 0:
                # Find the position of val in the goal state
                for gi in range(SIZE):
                    for gj in range(SIZE):
                        if goal[gi][gj] == val:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def find_blank(board):
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == 0:
                return i, j

def valid_moves(i, j):
    moves = []
    if i > 0: moves.append(('U', i - 1, j))
    if i < SIZE - 1: moves.append(('D', i + 1, j))
    if j > 0: moves.append(('L', i, j - 1))
    if j < SIZE - 1: moves.append(('R', i, j + 1))
    return moves

def apply_move(board, move):
    i, j = find_blank(board)
    direction, new_i, new_j = move
    new_board = [row[:] for row in board]
    new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
    return new_board

def a_star(start, goal):
    visited = set()
    heap = []
    heapq.heappush(heap, (manhattan(start, goal), 0, start, []))
    while heap:
        f, g, current, path = heapq.heappop(heap)
        if current == goal:
            return path + [current]
        visited.add(board_to_tuple(current))
        i, j = find_blank(current)
        for move in valid_moves(i, j):
            new_board = apply_move(current, move)
            if board_to_tuple(new_board) not in visited:
                heapq.heappush(heap, (g + 1 + manhattan(new_board, goal), g + 1, new_board, path + [current]))
    return []

def draw_board(board):
    return "\n".join([" ".join([str(cell) if cell != 0 else ' ' for cell in row]) for row in board])

def eight_puzzle_ui():
    st.set_page_config(page_title="8-Puzzle Solver", layout="centered")
    st.title("🧩 8 Puzzle Solver - A* Algorithm")
    st.markdown("Enter the initial and final board configurations (0 = blank):")
    
    # Input for initial state
    default_initial = "1 2 0\n3 4 5\n6 7 8"
    input_initial = st.text_area("Initial Puzzle Configuration", default_initial, height=100)
    
    # Input for final (goal) state
    default_goal = "1 2 3\n4 5 6\n7 8 0"
    input_goal = st.text_area("Final (Goal) Puzzle Configuration", default_goal, height=100)
    
    if st.button("Solve"):
        try:
            # Parse initial state
            start = [[int(x) for x in row.strip().split()] for row in input_initial.strip().split("\n")]
            if len(start) != SIZE or any(len(row) != SIZE for row in start):
                raise ValueError("Initial state must be a 3x3 grid.")
            
            # Parse goal state
            goal = [[int(x) for x in row.strip().split()] for row in input_goal.strip().split("\n")]
            if len(goal) != SIZE or any(len(row) != SIZE for row in goal):
                raise ValueError("Goal state must be a 3x3 grid.")
            
            # Validate that both states contain numbers 0-8 exactly once
            start_nums = sorted([num for row in start for num in row])
            goal_nums = sorted([num for row in goal for num in row])
            if start_nums != [0, 1, 2, 3, 4, 5, 6, 7, 8] or goal_nums != [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                raise ValueError("Both initial and goal states must contain exactly one of each number 0-8.")
            
            st.write("Solving...")
            solution_path = a_star(start, goal)
            if not solution_path:
                st.error("No solution found!")
            else:
                st.success(f"Solution found in {len(solution_path)-1} moves!")
                placeholder = st.empty()
                for step in solution_path:
                    placeholder.code(draw_board(step))
                    time.sleep(0.5)
        except Exception as e:
            st.error(f"Invalid input! Please use a 3x3 grid with numbers 0-8.\n\nError: {e}")

# ----------- Main App -----------
if __name__ == "__main__":
    eight_puzzle_ui()