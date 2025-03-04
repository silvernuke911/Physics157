import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the adjacency matrix
adj_matrix = np.array([
    [  0,  1,  0,  0,  1,1.4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
    [1.4,  1,  0,  1,  1,  0,  1,  0,1.4,  1,1.4,  0,  0,  0,  0,  0],
    [  0,  0,  1,1.4,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
    [  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,1.4,  1,  0,  0,  0,  0],
    [  0,  0,  0,  0,  1,1.4,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0],
    [  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,1.4,  1,  0,  0],
    [  0,  0,  0,  0,  0,1.4,  1,1.4,  0,  1,  0,  1,  0,  0,  1,1.4],
    [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  1,1.4,  0,  0,  0,  1,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1.4,  1,  0,  0,  1,  0]
])

import numpy as np
import heapq

def dijkstra(matrix, start, end):
    num_nodes = matrix.shape[0]
    distances = {i: float('inf') for i in range(num_nodes)}
    distances[start] = 0
    priority_queue = [(0, start)]
    previous_nodes = {i: None for i in range(num_nodes)}

    while priority_queue:
        curr_dist, curr_node = heapq.heappop(priority_queue)

        if curr_node == end:
            break  # Stop early if we reach the target node

        for neighbor, weight in enumerate(matrix[curr_node]):
            if weight > 0:  # Nonzero weight means there is an edge
                new_dist = curr_dist + weight
                if new_dist < distances[neighbor]:  # Found a shorter path
                    distances[neighbor] = new_dist
                    previous_nodes[neighbor] = curr_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    # Reconstruct the shortest path
    path = []
    node = end
    total_distance = distances[end]
    
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    
    path.reverse()
    
    return path, total_distance, len(path)

# Example usage
start_node = 0
end_node = 7

path, total_distance, num_nodes = dijkstra(adj_matrix, start_node, end_node)
print(f"Path: {path}")
print(f"Total Distance: {total_distance}")
print(f"Total Number of Nodes: {num_nodes}")

# Create graph
G = nx.Graph()

# Add nodes
num_nodes = len(adj_matrix)
G.add_nodes_from(range(num_nodes))

# Add edges with weights
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):  # Iterate upper triangle only (undirected graph)
        if adj_matrix[i, j] > 0:
            G.add_edge(i, j, weight=adj_matrix[i, j])

# Position nodes using spring layout
pos = nx.spring_layout(G, seed=42)

# Draw graph
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)

# Draw edges with weight labels
edge_labels = {(i, j): f'{adj_matrix[i, j]:.1f}' for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

plt.title("Graph Visualization of Adjacency Matrix")
plt.show()

