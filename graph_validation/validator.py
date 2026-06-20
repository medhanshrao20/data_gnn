import logging
import numpy as np
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger("pipeline")

class GraphValidator:
    @staticmethod
    def validate_graph(G: nx.Graph, graph_name: str) -> Dict[str, Any]:
        """
        Validates the structure of a graph and computes metrics:
        - Node count
        - Edge count
        - Density
        - Average degree
        - Connected components count
        - Isolated nodes count
        - Clustering coefficient
        - Graph diameter (if connected, or for largest component)
        - Graph radius (if connected, or for largest component)
        - Warning flags
        """
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Check directedness
        is_directed = G.is_directed()

        # Init default values
        density = 0.0
        avg_degree = 0.0
        num_components = 0
        num_isolated = 0
        avg_clustering = 0.0
        diameter = -1.0
        radius = -1.0
        
        warnings = []

        if num_nodes == 0:
            warnings.append("Empty Graph: No nodes present.")
            return {
                "node_count": 0, "edge_count": 0, "density": 0.0, "avg_degree": 0.0,
                "connected_components": 0, "isolated_nodes": 0, "clustering_coefficient": 0.0,
                "diameter": -1.0, "radius": -1.0, "warnings": warnings
            }

        # Degree calculation
        degrees = [d for n, d in G.degree()]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        
        # Density
        density = nx.density(G)
        
        # Isolated nodes
        num_isolated = len(list(nx.isolates(G)))
        
        # Connected components (for undirected) or weakly/strongly connected (for directed)
        if is_directed:
            undirected_G = G.to_undirected()
            num_components = nx.number_connected_components(undirected_G)
            components = list(nx.connected_components(undirected_G))
        else:
            num_components = nx.number_connected_components(G)
            components = list(nx.connected_components(G))

        # Clustering coefficient (average)
        try:
            # For directed graph, standard average_clustering handles it or converts it
            avg_clustering = nx.average_clustering(G)
        except Exception as e:
            logger.warning(f"Failed to compute clustering coefficient for {graph_name}: {e}")

        # Diameter and Radius
        # These metrics are only defined if the graph is fully connected.
        # Otherwise, compute on the largest connected component.
        try:
            if is_directed:
                is_conn = nx.is_strongly_connected(G)
            else:
                is_conn = nx.is_connected(G)
                
            if is_conn and num_nodes > 1:
                diameter = float(nx.diameter(G))
                radius = float(nx.radius(G))
            elif num_nodes > 1:
                warnings.append("Disconnected Graph: Diameter and radius computed on the largest component.")
                # Get largest component
                largest_cc = max(components, key=len)
                sub_g = G.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    diameter = float(nx.diameter(sub_g.to_undirected() if is_directed else sub_g))
                    radius = float(nx.radius(sub_g.to_undirected() if is_directed else sub_g))
        except Exception as e:
            logger.warning(f"Could not compute diameter/radius for {graph_name}: {e}")

        # Warnings logic
        if num_edges == 0:
            warnings.append("Degenerate Graph: No edges exist.")
        if num_nodes < 5:
            warnings.append(f"Small Graph Warning: Graph contains only {num_nodes} nodes.")
        if num_components > 1:
            warnings.append(f"Disconnected Graph: Contains {num_components} connected components.")
        if num_isolated > 0:
            warnings.append(f"Isolated Nodes Warning: {num_isolated} nodes have no connections.")

        stats = {
            "node_count": num_nodes,
            "edge_count": num_edges,
            "density": float(density),
            "avg_degree": float(avg_degree),
            "connected_components": num_components,
            "isolated_nodes": num_isolated,
            "clustering_coefficient": float(avg_clustering),
            "diameter": diameter,
            "radius": radius,
            "warnings": warnings
        }
        
        for warning in warnings:
            logger.warning(f"[{graph_name}] {warning}")

        return stats
