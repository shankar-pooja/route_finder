import networkx as nx
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

# ----------------------------
# Step 1: Define the graph
# ----------------------------
graph = {
    'A': {'B': 6, 'C': 3},
    'B': {'A': 6, 'C': 2, 'D': 5},
    'C': {'A': 3, 'B': 2, 'D': 3, 'E': 4},
    'D': {'B': 5, 'C': 3, 'E': 2, 'F': 3},
    'E': {'C': 4, 'D': 2, 'F': 5},
    'F': {'D': 3, 'E': 5}
}

# ----------------------------
# Step 2: Define heuristic for A*
# ----------------------------
heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 3, 'F': 0}

# ----------------------------
# Step 3: Breadth-First Search
# ----------------------------
def bfs(start, goal):
    visited = set()
    queue = [[start]]
    nodes_expanded = 0
    start_time = time.time()

    while queue:
        path = queue.pop(0)
        node = path[-1]
        nodes_expanded += 1

        if node == goal:
            return path, nodes_expanded, time.time() - start_time

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None, nodes_expanded, time.time() - start_time

# ----------------------------
# Step 4: Uniform Cost Search
# ----------------------------
def uniform_cost(start, goal):
    pq = PriorityQueue()
    pq.put((0, [start]))
    visited = set()
    nodes_expanded = 0
    start_time = time.time()

    while not pq.empty():
        cost, path = pq.get()
        node = path[-1]
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded, time.time() - start_time

        if node not in visited:
            visited.add(node)
            for neighbor, distance in graph[node].items():
                new_cost = cost + distance
                new_path = list(path)
                new_path.append(neighbor)
                pq.put((new_cost, new_path))

    return None, float('inf'), nodes_expanded, time.time() - start_time

# ----------------------------
# Step 5: A* Search
# ----------------------------
def a_star(start, goal):
    pq = PriorityQueue()
    pq.put((heuristic[start], 0, [start]))
    visited = set()
    nodes_expanded = 0
    start_time = time.time()

    while not pq.empty():
        est_total, cost, path = pq.get()
        node = path[-1]
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded, time.time() - start_time

        if node not in visited:
            visited.add(node)
            for neighbor, distance in graph[node].items():
                new_cost = cost + distance
                est = new_cost + heuristic[neighbor]
                new_path = list(path)
                new_path.append(neighbor)
                pq.put((est, new_cost, new_path))

    return None, float('inf'), nodes_expanded, time.time() - start_time

# ----------------------------
# Step 6: Visualization
# ----------------------------
def visualize(path, title):
    G = nx.Graph()
    for city, edges in graph.items():
        for neighbor, cost in edges.items():
            G.add_edge(city, neighbor, weight=cost)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=12, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight the found path
    if path:
        edge_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_path, edge_color="green", width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="orange")

    plt.title(title)
    plt.show()

# ----------------------------
# Step 7: Main Execution
# ----------------------------
start, goal = 'A', 'F'

# BFS
bfs_path, bfs_nodes, bfs_time = bfs(start, goal)
print(f"BFS Path: {bfs_path} | Nodes Expanded: {bfs_nodes} | Time: {bfs_time:.6f}s")
visualize(bfs_path, "BFS Path")

# UCS
ucs_path, ucs_cost, ucs_nodes, ucs_time = uniform_cost(start, goal)
print(f"UCS Path: {ucs_path} | Cost: {ucs_cost} | Nodes Expanded: {ucs_nodes} | Time: {ucs_time:.6f}s")
visualize(ucs_path, "Uniform Cost Search Path")

# A*
a_path, a_cost, a_nodes, a_time = a_star(start, goal)
print(f"A* Path: {a_path} | Cost: {a_cost} | Nodes Expanded: {a_nodes} | Time: {a_time:.6f}s")
visualize(a_path, "A* Search Path")

