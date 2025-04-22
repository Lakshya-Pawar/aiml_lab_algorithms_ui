import streamlit as st
import heapq
import time
import matplotlib.pyplot as plt

# Solver class (same logic as Tkinter version)
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


# Visualize the water levels
def draw_jugs(jug1_val, jug2_val, jug1_cap, jug2_cap):
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, max(jug1_cap, jug2_cap) + 1)

    # Jug 1
    ax.add_patch(plt.Rectangle((0.5, 0), 0.8, jug1_cap, fill=False, edgecolor="black", linewidth=1))
    ax.add_patch(plt.Rectangle((0.5, 0), 0.8, jug1_val, color="#00aaff"))
    ax.text(0.9, jug1_cap + 0.5, "Jug 1", ha="center", fontsize=12)
    ax.text(0.9, -1, f"{jug1_val}/{jug1_cap}", ha="center", fontsize=10, color="blue")

    # Jug 2
    ax.add_patch(plt.Rectangle((1.7, 0), 0.8, jug2_cap, fill=False, edgecolor="black", linewidth=1))
    ax.add_patch(plt.Rectangle((1.7, 0), 0.8, jug2_val, color="#00aaff"))
    ax.text(2.1, jug2_cap + 0.5, "Jug 2", ha="center", fontsize=12)
    ax.text(2.1, -1, f"{jug2_val}/{jug2_cap}", ha="center", fontsize=10, color="blue")

    ax.axis("off")
    st.pyplot(fig)


# Streamlit UI
st.set_page_config(page_title="Water Jug A* Visualizer", layout="centered")
st.title("Water Jug Problem - A* Algorithm")

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
