import os
import logging
import matplotlib
matplotlib.use('Agg')  # Headless mode for execution on servers/docker
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("pipeline")

class Visualizer:
    def __init__(self, station_name: str, results_dir: str):
        self.station_name = station_name
        self.results_dir = results_dir
        self.vis_dir = os.path.join(results_dir, station_name, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Style configurations
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })

    def plot_network(self, G: nx.Graph, output_name: str, title: str, is_weighted: bool = True):
        """Plots the network graph with node labels and weights."""
        if G.number_of_nodes() == 0:
            logger.warning(f"Skipping plot_network '{output_name}' because graph is empty.")
            return

        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Use circular layout for clean spacing with small (5-20) node counts
        pos = nx.circular_layout(G)
        
        # Nodes
        nx.draw_networkx_nodes(G, pos, node_color='#3498db', node_size=800, alpha=0.9, ax=ax)
        
        # Node Labels (Feature names)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="#2c3e50", ax=ax)

        # Edges and weights
        edges = G.edges(data=True)
        if len(edges) > 0:
            # Determine widths
            if is_weighted:
                weights = [d.get('weight', 1.0) for u, v, d in edges]
                # Scale weights for display thickness (min width 1, max 6)
                max_w = max(abs(w) for w in weights) if weights else 1.0
                max_w = max_w if max_w > 0 else 1.0
                widths = [max(1.0, min(6.0, (abs(d.get('weight', 1.0)) / max_w) * 5.0)) for u, v, d in edges]
                
                # Colors: blue for positive correlation, red for negative
                edge_colors = ['#2980b9' if d.get('weight', 0.0) >= 0 else '#c0392b' for u, v, d in edges]
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors, alpha=0.6, ax=ax)
                
                # Draw edge weight labels if not too cluttered
                if G.number_of_nodes() <= 12:
                    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.8, ax=ax)
            else:
                nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#7f8c8d', alpha=0.6, ax=ax)
        
        ax.set_title(f"{title} ({self.station_name})", pad=20)
        plt.tight_layout()
        
        save_path = os.path.join(self.vis_dir, f"{output_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved network plot to '{save_path}'.")

    def plot_heatmap(self, matrix: np.ndarray, labels: List[str], output_name: str, title: str):
        """Plots a correlation/similarity heatmap."""
        if matrix.size == 0 or len(labels) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use divergent colormap for correlation, sequential for similarity
        cmap = "RdBu_r" if "corr" in title.lower() or "weighted" in title.lower() else "viridis"
        vmin = -1.0 if "corr" in title.lower() or "weighted" in title.lower() else 0.0
        vmax = 1.0
        
        # Draw heatmap with annotations if nodes are small
        annot = len(labels) <= 15
        
        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"{title} - {self.station_name}", pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.vis_dir, f"{output_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved heatmap plot to '{save_path}'.")

    def plot_adjacency_grid(self, matrix: np.ndarray, labels: List[str], output_name: str, title: str):
        """Plots the adjacency matrix grid."""
        if matrix.size == 0 or len(labels) == 0:
            return

        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Binarize matrix for structure visualization
        binary_matrix = (np.abs(matrix) > 0.0).astype(float)
        
        sns.heatmap(
            binary_matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            cmap="binary",
            cbar=False,
            linewidths=0.5,
            linecolor="#bdc3c7",
            square=True,
            ax=ax
        )
        
        ax.set_title(f"{title} Structure - {self.station_name}", pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.vis_dir, f"{output_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved adjacency plot to '{save_path}'.")

    def plot_projection(self, coords: List[List[float]], labels: List[str], method: str, run_key: str):
        """Plots 2D projection of embeddings with annotated node names."""
        if not coords or len(coords) != len(labels):
            logger.warning(f"Invalid or missing coordinates for {method} projection (run: {run_key}). Skipping plot.")
            return

        pts = np.array(coords)
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Scatter points
        ax.scatter(pts[:, 0], pts[:, 1], s=120, c='#9b59b6', alpha=0.8, edgecolors='none')
        
        # Annotate labels
        for idx, label in enumerate(labels):
            ax.annotate(
                label,
                (pts[idx, 0], pts[idx, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=9,
                fontweight='semibold',
                color='#2c3e50'
            )
            
        ax.set_title(f"2D {method} Projection ({run_key})", pad=15)
        ax.set_xlabel(f"{method} Dim 1")
        ax.set_ylabel(f"{method} Dim 2")
        plt.tight_layout()
        
        save_path = os.path.join(self.vis_dir, f"{run_key}_{method.lower()}_projection.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved embedding projection plot to '{save_path}'.")
