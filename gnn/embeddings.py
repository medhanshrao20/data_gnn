import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger("pipeline")

# Try to import umap, fallback gracefully if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("umap-learn is not installed. UMAP projections will be skipped or fell back to t-SNE.")

class EmbeddingAnalyzer:
    def __init__(self, station_name: str, results_dir: str, settings: Any):
        self.station_name = station_name
        self.results_dir = results_dir
        self.settings = settings
        self.embed_dir = os.path.join(results_dir, station_name, "embeddings")
        self.reports_dir = os.path.join(results_dir, station_name, "reports")
        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def analyze_and_project(self) -> Dict[str, Any]:
        """
        Loads embeddings for all runs, computes structural statistics,
        and generates 2D coordinates for PCA, t-SNE, and UMAP.
        Returns:
            Dictionary of reports and coordinates.
        """
        results_report = {}
        
        # Discover embedding CSVs
        if not os.path.exists(self.embed_dir):
            logger.warning(f"Embeddings directory '{self.embed_dir}' does not exist. Skipping analysis.")
            return {}
            
        csv_files = [f for f in os.listdir(self.embed_dir) if f.endswith("_embeddings.csv")]
        
        if not csv_files:
            logger.warning("No embedding files found to analyze.")
            return {}

        logger.info(f"Analyzing {len(csv_files)} embedding configurations...")

        for file_name in csv_files:
            run_key = file_name.replace("_embeddings.csv", "")
            file_path = os.path.join(self.embed_dir, file_name)
            
            try:
                df = pd.read_csv(file_path, index_col=0)
                
                nodes = df.index.tolist()
                embeddings = df.values
                N, dim = embeddings.shape
                
                if N < 2:
                    logger.warning(f"Not enough nodes ({N}) in {run_key} to analyze embeddings.")
                    continue

                # 1. Quality metrics
                # Pairwise distances
                pairwise_dists = []
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.linalg.norm(embeddings[i] - embeddings[j])
                        pairwise_dists.append(dist)
                
                avg_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
                std_dist = float(np.std(pairwise_dists)) if pairwise_dists else 0.0
                
                # Dimensional variance
                dim_variance = float(np.mean(np.var(embeddings, axis=0)))
                # Sparsity (fraction of elements absolute val < 1e-3)
                sparsity = float(np.sum(np.abs(embeddings) < 1e-3) / (N * dim))
                
                metrics = {
                    "node_count": N,
                    "dimension": dim,
                    "avg_pairwise_distance": avg_dist,
                    "std_pairwise_distance": std_dist,
                    "dimension_variance": dim_variance,
                    "sparsity": sparsity
                }

                # 2. Dimensionality reduction (2D projections)
                projections = {}
                projection_errors = {}

                # A. PCA
                try:
                    # Cap components by N - 1 (since we have N samples)
                    pca_comps = min(2, dim, N - 1)
                    pca = PCA(n_components=pca_comps, random_state=self.settings.random_seed)
                    pca_coords = pca.fit_transform(embeddings)
                    # If dim was 1, pad to 2D
                    if pca_coords.shape[1] == 1:
                        pca_coords = np.hstack([pca_coords, np.zeros((N, 1))])
                    projections["PCA"] = pca_coords.tolist()
                except Exception as e:
                    logger.warning(f"PCA projection failed for {run_key}: {e}")
                    projection_errors["PCA"] = str(e)
                
                # B. t-SNE
                # Perplexity must be less than sample size
                perp = min(5.0, max(1.0, N - 1))
                try:
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perp,
                        random_state=self.settings.random_seed,
                        n_iter=1000,
                        init='random'
                    )
                    tsne_coords = tsne.fit_transform(embeddings)
                    projections["t-SNE"] = tsne_coords.tolist()
                except Exception as e:
                    logger.warning(f"t-SNE projection failed for {run_key}: {e}")
                    projection_errors["t-SNE"] = str(e)
                
                # C. UMAP
                if HAS_UMAP:
                    try:
                        n_neigh = min(5, max(2, N - 1))
                        reducer = umap.UMAP(
                            n_neighbors=n_neigh,
                            n_components=2,
                            random_state=self.settings.random_seed,
                            n_epochs=200
                        )
                        umap_coords = reducer.fit_transform(embeddings)
                        projections["UMAP"] = umap_coords.tolist()
                    except Exception as e:
                        logger.warning(f"UMAP projection failed for {run_key}: {e}")
                        projection_errors["UMAP"] = str(e)
                else:
                    # Fallback to copy of t-SNE or PCA
                    if "t-SNE" in projections:
                        projections["UMAP"] = projections["t-SNE"]
                        projection_errors["UMAP_fallback_source"] = "t-SNE"
                    elif "PCA" in projections:
                        projections["UMAP"] = projections["PCA"]
                        projection_errors["UMAP_fallback_source"] = "PCA"
                
                # Save results
                results_report[run_key] = {
                    "metrics": metrics,
                    "nodes": nodes,
                    "projections": projections,
                    "projection_errors": projection_errors
                }
                
                # Save coordinates to individual JSON
                coord_path = os.path.join(self.embed_dir, f"{run_key}_projections.json")
                tmp_coord_path = coord_path + ".tmp"
                with open(tmp_coord_path, "w") as f:
                    json.dump({
                        "run_key": run_key,
                        "nodes": nodes,
                        "metrics": metrics,
                        "projections": projections,
                        "projection_errors": projection_errors
                    }, f, indent=4)
                os.replace(tmp_coord_path, coord_path)
                    
            except Exception as e:
                logger.error(f"Error analyzing embedding file {file_name}: {e}", exc_info=True)
                continue

        # Save consolidating report
        consolidated_path = os.path.join(self.reports_dir, "embedding_analysis_report.json")
        tmp_consolidated_path = consolidated_path + ".tmp"
        with open(tmp_consolidated_path, "w") as f:
            json.dump(results_report, f, indent=4)
        os.replace(tmp_consolidated_path, consolidated_path)

        logger.info(f"Embedding analysis complete. Summary saved to '{consolidated_path}'.")
        return results_report
