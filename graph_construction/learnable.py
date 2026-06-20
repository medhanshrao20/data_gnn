import logging
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Any

logger = logging.getLogger("pipeline")

class LearnableGraphModel(nn.Module):
    def __init__(self, num_nodes: int, init_adj: np.ndarray = None):
        super(LearnableGraphModel, self).__init__()
        self.num_nodes = num_nodes
        
        # Initialize logits parameter
        if init_adj is not None:
            # Shift init_adj to parameter space to initialize near the correlations using arctanh
            # Clip between -1+epsilon and 1-epsilon
            eps = 1e-4
            clipped = np.clip(init_adj, -1 + eps, 1 - eps)
            param_init = np.arctanh(clipped)
            self.raw_adj = nn.Parameter(torch.FloatTensor(param_init))
        else:
            self.raw_adj = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)

    def forward(self) -> torch.Tensor:
        # Enforce symmetry
        adj = (self.raw_adj + self.raw_adj.t()) / 2.0
        # Convert parameters to range [-1, 1]
        adj = torch.tanh(adj)
        # Remove self loops
        mask = torch.eye(self.num_nodes, device=adj.device)
        adj = adj * (1.0 - mask)
        return adj


class LearnableGraphBuilder:
    def __init__(self, epochs: int = 150, lr: float = 0.01, weight_decay: float = 1e-4, l1_penalty: float = 0.005):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_penalty = l1_penalty

    def build_graph(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, nx.Graph, pd.DataFrame]:
        """
        Builds a Learnable Graph using PyTorch.
        Learns a symmetric adjacency matrix A to minimize reconstruction: MSE(X, X @ A) + L1_penalty * ||A||_1
        Returns:
            adjacency_matrix (np.ndarray): shape (N, N)
            graph (nx.Graph): NetworkX Graph object
            edge_list (pd.DataFrame): DataFrame with columns [source, target, weight]
        """
        N = len(feature_cols)
        adj_matrix = np.zeros((N, N))
        G = nx.Graph()
        
        for col in feature_cols:
            G.add_node(col)
            
        if N == 0:
            return adj_matrix, G, pd.DataFrame(columns=["source", "target", "weight"])

        # Compute initial correlation as starting point
        corr = df[feature_cols].corr().fillna(0).values
        # Strip self-loops for init
        np.fill_diagonal(corr, 0.0)

        # Prepare PyTorch data
        X = torch.FloatTensor(df[feature_cols].values)
        if torch.isnan(X).any() or torch.isinf(X).any():
            logger.warning("Learnable Graph: input contains NaN/inf values. Replacing with 0.0.")
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize model & optimizer
        model = LearnableGraphModel(N, init_adj=corr)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Keep track of best weights
        best_loss = float('inf')
        best_adj = corr.copy()

        logger.info(f"Training learnable graph adjacency matrix for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass: get adj matrix
            A = model()
            
            # Reconstruction X_hat = X @ A
            # (Batch, N) @ (N, N) -> (Batch, N)
            X_hat = torch.matmul(X, A)
            
            # Calculate loss
            recon_loss = nn.MSELoss()(X_hat, X)
            l1_loss = torch.sum(torch.abs(A))
            
            loss = recon_loss + self.l1_penalty * l1_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Learnable Graph: training loss is NaN at epoch {epoch}. Terminating optimization.")
                break
                
            loss.backward()
            optimizer.step()
            
            # Record best state
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_adj = A.detach().cpu().numpy()

        # Scale weights to [-1, 1] for normalization, keeping zero for weak connections
        final_adj = best_adj.copy()
        
        # Soft thresholding: remove very weak edges (e.g. < 0.05) to ensure sparsity
        final_adj[np.abs(final_adj) < 0.05] = 0.0
        
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                val = final_adj[i, j]
                if abs(val) > 0.0:
                    G.add_edge(feature_cols[i], feature_cols[j], weight=float(val))
                    edges.append({
                        "source": feature_cols[i],
                        "target": feature_cols[j],
                        "weight": float(val)
                    })
                    
        edge_list_df = pd.DataFrame(edges)
        if edge_list_df.empty:
            edge_list_df = pd.DataFrame(columns=["source", "target", "weight"])
            
        logger.info(f"Learned graph reconstruction completed (Best Loss: {best_loss:.6f}). Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}.")
        return final_adj, G, edge_list_df
