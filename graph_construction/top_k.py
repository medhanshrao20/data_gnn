import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("pipeline")

class TopKGraphBuilder:
    def __init__(self, k: int = 5, similarity_metric: str = "correlation"):
        self.k = k
        self.similarity_metric = similarity_metric  # 'correlation' or 'cosine'

    def build_graph(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, nx.DiGraph, pd.DataFrame]:
        """
        Builds a Top-K Similarity Graph (directed).
        For each node, we connect it to its K strongest neighbors.
        Returns:
            adjacency_matrix (np.ndarray): shape (N, N)
            graph (nx.DiGraph): NetworkX Directed Graph object
            edge_list (pd.DataFrame): DataFrame with columns [source, target, weight]
        """
        N = len(feature_cols)
        adj_matrix = np.zeros((N, N))
        G = nx.DiGraph()
        
        # Add all nodes
        for col in feature_cols:
            G.add_node(col)
            
        if N == 0:
            return adj_matrix, G, pd.DataFrame(columns=["source", "target", "weight"])

        # Compute similarity matrix
        if self.similarity_metric == "cosine":
            # shape (N, B) where B is rows
            features_matrix = df[feature_cols].values.T
            sim_matrix = cosine_similarity(features_matrix)
        else:
            # default to correlation
            sim_matrix = df[feature_cols].corr(method='pearson').fillna(0).values

        # Adjust K if N is too small
        actual_k = min(self.k, N - 1)
        if actual_k <= 0:
            logger.warning(f"Top-K graph: actual K adjusted to {actual_k} because N={N}.")
            return adj_matrix, G, pd.DataFrame(columns=["source", "target", "weight"])

        edges = []
        for i in range(N):
            # Sort neighbors by absolute similarity
            sims = sim_matrix[i].copy()
            sims[i] = 0.0  # exclude self-loop so abs is 0
            
            # Get indices of top K largest absolute similarity values
            top_k_indices = np.argsort(np.abs(sims))[-actual_k:]
            
            for target_idx in top_k_indices:
                val = sim_matrix[i, target_idx]
                # Filter edges below a minimum absolute similarity threshold (e.g., zero-similarity)
                if abs(val) > 1e-6:
                    adj_matrix[i, target_idx] = val
                    G.add_edge(feature_cols[i], feature_cols[target_idx], weight=float(val))
                    edges.append({
                        "source": feature_cols[i],
                        "target": feature_cols[target_idx],
                        "weight": float(val)
                    })

        edge_list_df = pd.DataFrame(edges)
        if edge_list_df.empty:
            edge_list_df = pd.DataFrame(columns=["source", "target", "weight"])
            
        logger.info(f"Built Top-{actual_k} Graph ({self.similarity_metric}) with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return adj_matrix, G, edge_list_df
