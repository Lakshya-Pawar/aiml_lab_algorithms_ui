import streamlit as st
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# ----------- Water Jug Solver -----------
class WaterJugSolver:
    def __init__(self, jug1, jug2, goal):
        self.jug1 = jug1
        self.jug2 = jug2
        self.goal = goal
        self.visited = set()

    def is_goal(self, state):
        return self.goal in state

    def get_heuristic(self, state):
        return min(abs(state[0] - self.goal), abs(state[1] - self.goal))

    def get_successors(self, state):
        successors = []
        a, b = state
        j1, j2 = self.jug1, self.jug2
        successors.append(((j1, b), "Fill Jug 1"))
        successors.append(((a, j2), "Fill Jug 2"))
        successors.append(((0, b), "Empty Jug 1"))
        successors.append(((a, 0), "Empty Jug 2"))
        transfer = min(a, j2 - b)
        successors.append(((a - transfer, b + transfer), "Pour Jug 1 → Jug 2"))
        transfer = min(b, j1 - a)
        successors.append(((a + transfer, b - transfer), "Pour Jug 2 → Jug 1"))
        return successors

    def solve(self):
        heap = []
        heapq.heappush(heap, (0, 0, (0, 0), []))
        while heap:
            f, g, current, path = heapq.heappop(heap)
            if self.is_goal(current):
                return path + [(current, "Goal Reached")]
            if current in self.visited:
                continue
            self.visited.add(current)
            for neighbor, action in self.get_successors(current):
                if neighbor not in self.visited:
                    new_path = path + [(current, action)]
                    new_g = g + 1
                    new_f = new_g + self.get_heuristic(neighbor)
                    heapq.heappush(heap, (new_f, new_g, neighbor, new_path))
        return None

def draw_jugs(jug1_val, jug2_val, jug1_cap, jug2_cap):
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, max(jug1_cap, jug2_cap) + 1)
    ax.add_patch(plt.Rectangle((0.5, 0), 0.8, jug1_cap, fill=False, edgecolor="black", linewidth=1))
    ax.add_patch(plt.Rectangle((0.5, 0), 0.8, jug1_val, color="#00aaff"))
    ax.text(0.9, jug1_cap + 0.5, "Jug 1", ha="center", fontsize=12)
    ax.text(0.9, -1, f"{jug1_val}/{jug1_cap}", ha="center", fontsize=10, color="blue")
    ax.add_patch(plt.Rectangle((1.7, 0), 0.8, jug2_cap, fill=False, edgecolor="black", linewidth=1))
    ax.add_patch(plt.Rectangle((1.7, 0), 0.8, jug2_val, color="#00aaff"))
    ax.text(2.1, jug2_cap + 0.5, "Jug 2", ha="center", fontsize=12)
    ax.text(2.1, -1, f"{jug2_val}/{jug2_cap}", ha="center", fontsize=10, color="blue")
    ax.axis("off")
    st.pyplot(fig)

def water_jug_ui():
    st.title("💧 Water Jug Problem - A* Algorithm")
    with st.form("jug_form"):
        jug1 = st.number_input("Enter Jug 1 Capacity:", min_value=1, value=4)
        jug2 = st.number_input("Enter Jug 2 Capacity:", min_value=1, value=3)
        goal = st.number_input("Enter Goal Amount:", min_value=1, value=2)
        start = st.form_submit_button("Solve & Visualize")
    if start:
        solver = WaterJugSolver(jug1, jug2, goal)
        solution = solver.solve()
        if not solution:
            st.error("❌ No solution found.")
        else:
            st.success("✅ Solution found!")
            placeholder = st.empty()
            for i, (state, action) in enumerate(solution):
                with placeholder.container():
                    st.subheader(f"Step {i+1}: {action}")
                    draw_jugs(state[0], state[1], jug1, jug2)
                    st.markdown("---")
                time.sleep(1.2)

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
    st.title("🧩 8 Puzzle Solver - A* Algorithm")
    st.markdown("Enter the initial and final board configurations (0 = blank):")
    
    # Input for initial state
    default_initial = "1 2 0\n3 4 5\n6 7 8"
    input_initial = st.text_area("Initial Puzzle State", default_initial, height=100)
    
    # Input for final (goal) state
    default_goal = "1 2 3\n4 5 6\n7 8 0"
    input_goal = st.text_area("Final (Goal) Puzzle State", default_goal, height=100)
    
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

# ----------- TSP Solver -----------
def tsp_dfs(graph, start):
    n = len(graph)
    visited = [False] * n
    min_cost = float('inf')
    best_path = []
    def dfs(curr, count, cost, path):
        nonlocal min_cost, best_path
        if count == n and graph[curr][start]:
            total_cost = cost + graph[curr][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            return
        for i in range(n):
            if not visited[i] and graph[curr][i]:
                visited[i] = True
                dfs(i, count + 1, cost + graph[curr][i], path + [i])
                visited[i] = False
    visited[start] = True
    dfs(start, 1, 0, [start])
    return min_cost, best_path

def tsp_bfs(graph, start):
    n = len(graph)
    queue = deque()
    min_cost = float('inf')
    best_path = []
    queue.append((start, [start], 0))
    while queue:
        node, path, cost = queue.popleft()
        if len(path) == n:
            if graph[node][start] != 0:
                total_cost = cost + graph[node][start]
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = path + [start]
            continue
        for i in range(n):
            if i not in path and graph[node][i] != 0:
                queue.append((i, path + [i], cost + graph[node][i]))
    return min_cost, best_path

def tsp_ui():
    st.title("🧭 TSP using BFS and DFS")
    algo = st.radio("Choose Algorithm", ["DFS", "BFS"])
    n = 4
    prefill = ["0 10 15 20", "10 0 35 25", "15 35 0 30", "20 25 30 0"]
    st.write("### Distance Matrix:")
    matrix_input = []
    valid_input = True
    for i in range(n):
        row = st.text_input(f"Row {i+1}:", value=prefill[i])
        try:
            values = list(map(int, row.strip().split()))
            if len(values) == n:
                matrix_input.append(values)
            else:
                valid_input = False
                st.error(f"Row {i+1} must have exactly {n} numbers.")
        except ValueError:
            valid_input = False
            st.error(f"Invalid values in Row {i+1}.")
    if valid_input and st.button("Solve"):
        if algo == "DFS":
            cost, path = tsp_dfs(matrix_input, 0)
        else:
            cost, path = tsp_bfs(matrix_input, 0)
        if cost == float('inf'):
            st.error("No valid path found.")
        else:
            st.success(f"✅ Minimum Cost: {cost}")
            st.write("📍 Optimal Path:", " ➝ ".join(map(str, path)))

# ----------- Find-S Algorithm -----------
def find_s(examples):
    hypothesis = ['0'] * len(examples[0][0])
    for example, label in examples:
        if label == 1:
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    return hypothesis

def find_s_ui():
    st.title("🧠 Find-S Algorithm")
    num_attributes = st.number_input("Number of attributes:", min_value=1, step=1)
    attribute_names = [st.text_input(f"Attribute {i+1}", value=f"Attribute {i+1}") for i in range(num_attributes)]
    st.subheader("Add Examples")
    if 'examples' not in st.session_state:
        st.session_state.examples = []
    with st.form("example_form"):
        example_values = [st.text_input(f"{attribute_names[i]}", key=f"val_{i}") for i in range(num_attributes)]
        label = st.selectbox("Label", [1, 0], format_func=lambda x: "Positive" if x == 1 else "Negative")
        if st.form_submit_button("Add Example"):
            if all(value.strip() for value in example_values):
                st.session_state.examples.append((example_values, label))
                st.success("Example added!")
            else:
                st.error("Fill all fields.")
    if st.session_state.examples:
        st.subheader("Examples")
        for i, (example, label) in enumerate(st.session_state.examples):
            st.write(f"Example {i+1}: {example}, Label: {'Positive' if label == 1 else 'Negative'}")
    if st.button("Compute Hypothesis"):
        if st.session_state.examples:
            hypothesis = find_s(st.session_state.examples)
            st.subheader("Hypothesis")
            st.write({attribute_names[i]: hypothesis[i] for i in range(len(hypothesis))})
        else:
            st.warning("Add at least one example.")

# ----------- Main App -----------
st.set_page_config(page_title="Algorithm Visualizer", layout="centered")

# Custom CSS for sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
        padding: 20px;
        border-right: 2px solid #ff4d94;
    }
    .sidebar .sidebar-content .stButton>button {
        background-color: #16213e;
        color: white;
        border: 2px solid #ff4d94;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 10px 0;
        width: 100%;
        text-align: left;
        font-size: 16px;
        transition: all 0.3s;
    }
    .sidebar .sidebar-content .stButton>button:hover {
        background-color: #ff4d94;
        color: white;
    }
    .sidebar .sidebar-content .stButton>button[selected] {
        background-color: #ff4d94;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content h1 {
        color: #ff4d94;
        font-size: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content hr {
        border: 1px solid #ff4d94;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Algorithm Visualizer")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Initialize session state for app_mode if not present
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = None

# Button navigation with session state and icons
if st.sidebar.button("💧 Water Jug (A*)"):
    st.session_state.app_mode = "Water Jug (A*)"
if st.sidebar.button("🧩 8-Puzzle (A*)"):
    st.session_state.app_mode = "8-Puzzle (A*)"
if st.sidebar.button("🗺️ TSP (BFS/DFS)"):
    st.session_state.app_mode = "TSP (BFS/DFS)"
if st.sidebar.button("🧠 Find-S"):
    st.session_state.app_mode = "Find-S"

# Display the selected UI based on session state
if st.session_state.app_mode:
    if st.session_state.app_mode == "Water Jug (A*)":
        water_jug_ui()
    elif st.session_state.app_mode == "8-Puzzle (A*)":
        eight_puzzle_ui()
    elif st.session_state.app_mode == "TSP (BFS/DFS)":
        tsp_ui()
    elif st.session_state.app_mode == "Find-S":
        find_s_ui()
else:
    st.write("Please select an algorithm from the sidebar.")