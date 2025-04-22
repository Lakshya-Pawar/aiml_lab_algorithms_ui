import streamlit as st
from collections import deque

# ----------- DFS Function -----------

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

# ----------- BFS Function -----------

def tsp_bfs(graph, start):
    n = len(graph)
    queue = deque()
    min_cost = float('inf')
    best_path = []

    queue.append((start, [start], 0))

    while queue:
        node, path, cost = queue.popleft()

        if len(path) == n and graph[node][start] != 0:
            total_cost = cost + graph[node][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            continue

        for i in range(n):
            if i not in path and graph[node][i] != 0:
                queue.append((i, path + [i], cost + graph[node][i]))

    return min_cost, best_path

# ----------- Streamlit UI -----------

st.title("🧭 TSP using BFS and DFS")
algo = st.radio("Choose Algorithm", ["DFS", "BFS"])
n = 4  # fixed matrix size

# Prefilled distance matrix
prefill = [
    "0 10 15 20",
    "10 0 35 25",
    "15 35 0 30",
    "20 25 30 0"
]

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
