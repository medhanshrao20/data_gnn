import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
from sklearn.decomposition import PCA
from .models import GATModel, GraphSAGEModel

logger = logging.getLogger("pipeline")

class GNNTrainer:
    def __init__(self, station_name: str, results_dir: str, settings: Any):
        self.station_name = station_name
        self.results_dir = results_dir
        self.settings = settings
        self.embed_dir = os.path.join(results_dir, station_name, "embeddings")
        self.reports_dir = os.path.join(results_dir, station_name, "reports")
        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def run_gnn_pipeline(self, scaled_df: pd.DataFrame, feature_cols: List[str], graphs_dict: Dict[str, Tuple[np.ndarray, Any]]) -> Dict[str, Any]:
        """
        Runs the GNN learning stage for a station.
        Input:
            scaled_df (pd.DataFrame): scaled dataset
            feature_cols (List[str]): feature names (representing nodes)
            graphs_dict (Dict): maps graph_type ('correlation', 'top_k', 'weighted', 'learnable')
                                to Tuple (adjacency_matrix, networkx_graph)
        Returns:
            Dictionary containing GNN training logs, loss metadata, and paths to embeddings.
        """
        N = len(feature_cols)
        pipeline_report = {
            "station": self.station_name,
            "node_count": N,
            "skipped": False,
            "runs": {}
        }

        # Small Graph Safety Rules
        if N < 3:
            msg = f"Skipping GNN Stage: Feature graph has only {N} nodes (minimum 3 required)."
            logger.warning(msg)
            pipeline_report["skipped"] = True
            pipeline_report["reason"] = msg
            # Save explanation report
            explanation_path = os.path.join(self.reports_dir, "gnn_explanation.json")
            tmp_explanation_path = explanation_path + ".tmp"
            with open(tmp_explanation_path, "w") as f:
                json.dump(pipeline_report, f, indent=4)
            os.replace(tmp_explanation_path, explanation_path)
            return pipeline_report

        if N < 5:
            logger.warning(f"Small Graph Warning: Graph has only {N} nodes (less than 5). Continuing processing.")
        elif N < 10:
            logger.warning(f"Small Graph Warning: Graph has only {N} nodes (less than 10). Continuing processing.")

        # 1. Node Feature Construction (transpose and reduce B dimension via PCA)
        # Raw features shape (B, N) where B = rows
        X_raw = scaled_df[feature_cols].values.T  # (N, B)
        
        # Memory safety: cap the number of rows (B) to avoid PCA memory explosion
        max_rows = 10000
        if X_raw.shape[1] > max_rows:
            logger.info(f"Subsampling raw features from {X_raw.shape[1]} to {max_rows} rows to prevent OOM.")
            # Deterministic per-station sampling
            station_hash = sum(ord(c) for c in self.station_name)
            rng = np.random.default_rng(self.settings.random_seed + station_hash)
            indices = rng.choice(X_raw.shape[1], max_rows, replace=False)
            X_raw = X_raw[:, indices]
            
        B = X_raw.shape[1]
        
        # Determine PCA components (target projection dimension D)
        # Note: PCA n_components cannot exceed the number of samples minus 1 (which is N - 1)
        D = min(N - 1, B, 32)
        if B > D:
            logger.info(f"Reducing node raw feature series length from B={B} to D={D} using PCA.")
            pca = PCA(n_components=D, random_state=self.settings.random_seed)
            X_node = pca.fit_transform(X_raw)  # (N, D)
        else:
            X_node = X_raw.copy()
            D = B

        # Prepare PyTorch Tensors
        X_tensor = torch.FloatTensor(X_node)
        
        # Track training summaries
        for graph_type, (adj_matrix, G_nx) in graphs_dict.items():
            # Preserve signed edge weights. GraphSAGE normalizes by absolute degree,
            # while GAT uses non-zero signed edges as the attention mask.
            safe_adj_matrix = np.nan_to_num(adj_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            adj_tensor = torch.FloatTensor(safe_adj_matrix)
            
            # Train GAT & GraphSAGE
            for model_name in ["GAT", "GraphSAGE"]:
                run_key = f"{graph_type}_{model_name}"
                logger.info(f"Training {model_name} on {graph_type} graph...")
                
                # Instantiating Model & Decoder
                hidden_dim = self.settings.gnn_hidden_dim
                embed_dim = self.settings.gnn_embedding_dim
                
                # Check model dims relative to N
                if N < embed_dim:
                    # Adjust embedding dimension for tiny graphs to avoid dimensional issues
                    embed_dim = max(2, N - 1)
                    hidden_dim = max(4, embed_dim * 2)
                
                if model_name == "GAT":
                    gnn_model = GATModel(in_features=D, hidden_dim=hidden_dim, out_features=embed_dim)
                else:
                    gnn_model = GraphSAGEModel(in_features=D, hidden_dim=hidden_dim, out_features=embed_dim)
                
                # Reconstruction Decoder
                decoder = nn.Linear(embed_dim, D)
                
                # Optimizer
                optimizer = optim.Adam(
                    list(gnn_model.parameters()) + list(decoder.parameters()), 
                    lr=self.settings.gnn_lr, 
                    weight_decay=self.settings.gnn_weight_decay
                )
                criterion = nn.MSELoss()

                # Train
                best_loss = float('inf')
                best_embeddings = None
                stable = True
                
                for epoch in range(self.settings.gnn_epochs):
                    gnn_model.train()
                    decoder.train()
                    optimizer.zero_grad()
                    
                    # Forward pass
                    z = gnn_model(X_tensor, adj_tensor)
                    X_recon = decoder(z)
                    
                    loss = criterion(X_recon, X_tensor)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"Numerical explosion detected in GNN {run_key} at epoch {epoch}. Terminating.")
                        stable = False
                        break
                        
                    loss.backward()
                    optimizer.step()
                    
                    loss_val = loss.item()
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_embeddings = z.detach().cpu().numpy()

                # Embedding Validation
                if not stable or best_embeddings is None:
                    logger.warning(f"GNN run {run_key} failed or was unstable.")
                    best_embeddings = np.zeros((N, embed_dim))
                    validation_status = "Unstable / Exploded"
                else:
                    # Check for collapsed embeddings (all identical or zero variance)
                    std_devs = np.std(best_embeddings, axis=0)
                    if np.all(std_devs < 1e-4):
                        logger.warning(f"Collapsed Embeddings Warning: {run_key} embeddings have collapsed (zero variance).")
                        validation_status = "Collapsed (Zero Variance)"
                    elif np.isnan(best_embeddings).any() or np.isinf(best_embeddings).any():
                        logger.warning(f"Embeddings Warning: {run_key} contains NaN/inf values.")
                        best_embeddings = np.nan_to_num(best_embeddings)
                        validation_status = "NaNs present (cleared)"
                    else:
                        validation_status = "Stable"

                final_loss_val = float(best_loss)
                if np.isinf(final_loss_val) or validation_status in ["Unstable / Exploded", "Collapsed (Zero Variance)"]:
                    final_loss_val = "failed"
                    validation_status = "Failed"
                    logger.warning(f"GNN run {run_key} failed. Not saving embeddings.")
                    # Log report and skip saving invalid embeddings
                    pipeline_report["runs"][run_key] = {
                        "graph_type": graph_type,
                        "model_name": model_name,
                        "final_loss": final_loss_val,
                        "validation": validation_status,
                        "embedding_path": None
                    }
                    continue
                    
                embed_df = pd.DataFrame(best_embeddings, index=feature_cols)
                
                # Use atomic write for artifact creation
                embed_csv_path = os.path.join(self.embed_dir, f"{run_key}_embeddings.csv")
                tmp_path = embed_csv_path + ".tmp"
                embed_df.to_csv(tmp_path)
                os.replace(tmp_path, embed_csv_path)
                    
                # Log report
                pipeline_report["runs"][run_key] = {
                    "graph_type": graph_type,
                    "model_name": model_name,
                    "final_loss": final_loss_val,
                    "validation": validation_status,
                    "embedding_path": embed_csv_path
                }
                
                logger.info(f"Saved {run_key} embeddings to '{embed_csv_path}'. Status: {validation_status}")

        # Save orchestrator run report
        report_path = os.path.join(self.reports_dir, "gnn_report.json")
        tmp_report_path = report_path + ".tmp"
        with open(tmp_report_path, "w") as f:
            json.dump(pipeline_report, f, indent=4)
        os.replace(tmp_report_path, report_path)

        return pipeline_report
