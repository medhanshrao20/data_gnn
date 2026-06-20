import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Any

logger = logging.getLogger("pipeline")

class WeightedGraphBuilder:
    def build_graph(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, nx.Graph, pd.DataFrame]:
        """
        Builds a fully connected Weighted Graph.
        All pairwise correlations (except self) are preserved as edges.
        Returns:
            adjacency_matrix (np.ndarray): shape (N, N)
            graph (nx.Graph): NetworkX Graph object
            edge_list (pd.DataFrame): DataFrame with columns [source, target, weight]
        """
        N = len(feature_cols)
        adj_matrix = np.zeros((N, N))
        G = nx.Graph()
        
        # Add all nodes
        for col in feature_cols:
            G.add_node(col)
            
        if N == 0:
            return adj_matrix, G, pd.DataFrame(columns=["source", "target", "weight"])

        # Compute correlation matrix
        corr_matrix = df[feature_cols].corr(method='pearson').fillna(0).values
        
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                val = corr_matrix[i, j]
                if val != 0.0:
                    adj_matrix[i, j] = val
                    adj_matrix[j, i] = val
                    G.add_edge(feature_cols[i], feature_cols[j], weight=float(val))
                    edges.append({
                        "source": feature_cols[i],
                        "target": feature_cols[j],
                        "weight": float(val)
                    })

        edge_list_df = pd.DataFrame(edges)
        if edge_list_df.empty:
            edge_list_df = pd.DataFrame(columns=["source", "target", "weight"])
            
        logger.info(f"Built Weighted Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return adj_matrix, G, edge_list_df
