import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger("pipeline")

class CentralityAnalyzer:
    @staticmethod
    def compute_centralities(G: nx.Graph, graph_name: str) -> Dict[str, Dict[str, float]]:
        """
        Computes centralities for each node in the graph:
        - Degree Centrality
        - Betweenness Centrality
        - Closeness Centrality
        - Eigenvector Centrality (with convergence safety)
        - PageRank (with convergence safety)
        Returns:
            Dictionary mapping node names to their centrality scores.
        """
        nodes = list(G.nodes())
        
        # Initialize results structure
        results = {node: {
            "degree_centrality": 0.0,
            "betweenness_centrality": 0.0,
            "closeness_centrality": 0.0,
            "eigenvector_centrality": 0.0,
            "pagerank": 0.0
        } for node in nodes}

        if len(nodes) == 0:
            return results

        has_weights = any('weight' in d for u,v,d in G.edges(data=True))
        if has_weights:
            for u, v, d in G.edges(data=True):
                d['abs_weight'] = abs(d.get('weight', 1.0))
                d['distance'] = 1.0 / (d['abs_weight'] + 1e-6)

        # 1. Degree Centrality
        try:
            deg = nx.degree_centrality(G)
            for node, val in deg.items():
                results[node]["degree_centrality"] = float(val)
        except Exception as e:
            logger.warning(f"Degree centrality calculation failed for {graph_name}: {e}")

        # 2. Betweenness Centrality
        try:
            bet = nx.betweenness_centrality(G, weight='distance' if has_weights else None)
            for node, val in bet.items():
                results[node]["betweenness_centrality"] = float(val)
        except Exception as e:
            logger.warning(f"Betweenness centrality calculation failed for {graph_name}: {e}")

        # 3. Closeness Centrality
        try:
            close = nx.closeness_centrality(G, distance='distance' if has_weights else None)
            for node, val in close.items():
                results[node]["closeness_centrality"] = float(val)
        except Exception as e:
            logger.warning(f"Closeness centrality calculation failed for {graph_name}: {e}")

        # 4. Eigenvector Centrality
        try:
            # Add tolerance and max_iter for stability
            eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-5, weight='abs_weight' if has_weights else None)
            for node, val in eig.items():
                results[node]["eigenvector_centrality"] = float(val)
        except nx.PowerIterationFailedConvergence:
            logger.warning(f"Eigenvector centrality failed to converge for {graph_name}. Filling with 0.0.")
        except Exception as e:
            logger.warning(f"Eigenvector centrality calculation failed for {graph_name}: {e}")

        # 5. PageRank
        try:
            pr = nx.pagerank(G, max_iter=1000, tol=1e-5, weight='abs_weight' if has_weights else None)
            for node, val in pr.items():
                results[node]["pagerank"] = float(val)
        except nx.PowerIterationFailedConvergence:
            logger.warning(f"PageRank failed to converge for {graph_name}. Filling with 0.0.")
        except Exception as e:
            logger.warning(f"PageRank calculation failed for {graph_name}: {e}")

        return results
