import torch
import networkx as nx
import matplotlib.pyplot as plt

# Define all gates with their adjacency matrices and example weights
gates = {
    "FALSE": torch.zeros((6,6), dtype=torch.int32),
    "TRUE": torch.zeros((6,6), dtype=torch.int32),
    "AND": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "OR": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "NAND": torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "NOR": torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
    "XOR": torch.tensor([
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.int32),
}

# Example: uniform dummy weights for visualization

node_labels = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H'}
func_colors = ['#fe9a99', '#c2ebc1', '#fed899', '#bfdbe8', '#d5bfaf', '#aeb1d7', '#f8e7ec']

# Plot all gates
fig, axes = plt.subplots(3, 4, figsize=(15,5))
axes = axes.flatten()

for i, (name, adj) in enumerate(gates.items()):
    w, h = adj.size()
    dummy_weights = torch.ones((w, h)) * 10

    G = nx.DiGraph()
    # Add nodes
    for n in range(w):
        G.add_node(n, label=node_labels[n])
    # Add edges
    rows, cols = torch.nonzero(adj, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_weights = [dummy_weights[r,c].item() for r,c in edges]
    edge_widths = [(w / max(edge_weights) * 5) if edge_weights else 1 for w in edge_weights]
    edge_colors = ['green']*len(edges)  # just uniform for simplicity
    
    for (r,c) in edges:
        G.add_edge(r,c)
    
    # Node activity
    node_activity = adj.sum(dim=1).tolist()
    node_sizes = [50 + a*20 for a in node_activity]
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=node_sizes,
            node_color=func_colors[i % len(func_colors)], width=edge_widths, edge_color=edge_colors,
            arrowsize=20, ax=axes[i])
    axes[i].set_title(name)

# Hide unused subplot if any
for j in range(i+1, len(axes)): # type: ignore
    axes[j].axis('off')

plt.suptitle("Minimal DAGs for Basic Boolean Gates")
plt.show()
