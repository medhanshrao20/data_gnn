import os
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize a saved graph from the results folder.")
    parser.add_argument(
        "--file", 
        type=str, 
        default="results/SNORRE_B/graphs/correlation_adjacency.csv", 
        help="Path to the saved adjacency.csv file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        print("Please make sure you have run the pipeline and the path is correct.")
        return

    # 1. Load the adjacency CSV
    df = pd.read_csv(args.file, index_col=0)
    print(f"Loading graph from {args.file}...")
    
    # 2. Build the NetworkX Graph from adjacency matrix
    G = nx.from_pandas_adjacency(df)

    # 3. Plot the Graph
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(G)  # Arrange nodes in a ring layout

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color='#3498db', node_size=1200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="#2c3e50")

    # Draw Edges (thickness scaled by connection strength)
    edges = G.edges(data=True)
    weights = [abs(d['weight']) for u, v, d in edges]
    max_weight = max(weights) if weights else 1.0
    widths = [max(1.0, (w / max_weight) * 5.0) for w in weights]
    edge_colors = ['#2980b9' if d['weight'] >= 0 else '#c0392b' for u, v, d in edges]

    nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors, alpha=0.6)
    
    # Draw edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # 4. Show on screen
    graph_name = os.path.basename(args.file).replace("_edgelist.csv", "").capitalize()
    plt.title(f"Interactive Visualization: {graph_name} Graph", pad=20, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
