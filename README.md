# Industrial-Grade Autonomous Graph Analytics Framework (Phase 1)

An autonomous, schema-agnostic, dataset-tolerant machine learning framework that automatically discovers tabular datasets, profiles and cleans them, constructs four distinct feature-level relationship graph topologies, validates structural connections, and trains Graph Neural Networks (GAT & GraphSAGE) to produce premium, low-dimensional feature embeddings.

---

## 🌟 Key Architectural Strengths

- **Station & Dataset Agnostic**: No station names, feature dimensions, schemas, chronological intervals, or column configurations are hardcoded. Everything is discovered and processed adaptively.
- **Pure PyTorch GNNs**: Bypasses system-specific compilation failures of `torch-geometric` by utilizing custom vectorized Graph Attention (GAT) and GraphSAGE layers in pure PyTorch.
- **Fail-Safe Integrity**: Catches exceptions at individual stations, files, or iterations. Schema drifts, NaN values, corrupt formats, or singular features will trigger detailed error reports/warnings but will **never** crash the global pipeline.
- **Multiprocessing Ready**: Processes independent station folders concurrently to minimize execution runtime.
- **Aesthetic Visualizations**: Generates layout-optimized network plots, correlation heatmaps, connection grids, and embedding projections (PCA, t-SNE, and UMAP).

---

## 🚀 Execution Guide

### Method 1: Local Shell (Windows PowerShell or Bash)
1. **Create and Activate Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate node_gnn
   # OR
   pip install -r requirements.txt
   ```
2. **Populate Datasets**:
   Place folders (representing stations) with tabular files (CSV, Parquet, TSV, or Excel) inside the `data/` folder.
3. **Run Pipeline**:
   ```bash
   python main.py
   ```

### Method 2: Docker Setup
Build and run the container, mounting local data and results:
```bash
# Build
docker build -t node-gnn .

# Run
docker run -it --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" node-gnn
```

---

## 🛠️ CLI Interface Options

Override `config.yaml` parameters using CLI arguments:
```bash
python main.py --config custom_config.yaml --seed 100 --parallel false --log_level DEBUG
```
- `--config`: Custom yaml configuration.
- `--data_dir`: Override source data folder (defaults to `data`).
- `--results_dir`: Override outcome results folder (defaults to `results`).
- `--seed`: Override framework random seed.
- `--parallel`: `true` or `false` to toggle multiprocessing.
- `--workers`: Force a specific number of parallel processes.
- `--log_level`: Change console detail level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

---

## 📁 Pipeline Phases & Outputs

For every station (e.g. `Station_A`), the engine creates a directory under `results/Station_A/` with this structure:

```
results/
├── Station_A/
│   ├── audit/
│   │   ├── audit_report.json            # Statistics before preprocessing (uniqueness, types, memory)
│   │   └── audit_report.md              # Human-readable data profiling markdown table
│   ├── cleaned/
│   │   └── cleaned.csv                  # Merged, imputed, outlier-treated, deduplicated dataset
│   ├── scaled/
│   │   └── scaled.csv                   # Cleaned dataset with features scaled by StandardScaler
│   ├── graphs/
│   │   ├── correlation_adjacency.csv   # Adjacency matrices for Correlation Graph
│   │   ├── correlation_edgelist.csv     # Sorted edge connections list with weight values
│   │   ├── [top_k, weighted, learnable]_adjacency.csv
│   │   └── [top_k, weighted, learnable]_edgelist.csv
│   ├── embeddings/
│   │   ├── correlation_GAT_embeddings.csv  # 8-dim learned node embedding files
│   │   ├── correlation_GAT_projections.json # PCA/t-SNE/UMAP 2D projection coordinates
│   │   └── [top_k, weighted, learnable]_[GAT, GraphSAGE]_embeddings/projections
│   ├── visualizations/
│   │   ├── correlation_network.png      # Networkx graph nodes/edges labeled with feature names
│   │   ├── correlation_heatmap.png      # Adjacent heatmap visualizations
│   │   ├── correlation_adj_grid.png     # Grid adjacency connection grids
│   │   └── [run_key]_[pca, tsne, umap]_projection.png # 2D visual scatterplots of embeddings
│   ├── logs/
│   │   └── station.log                  # Dedicated debug/trace file for this station
│   └── reports/
│       ├── cleaning_report.json         # Outliers and columns dropped tracking
│       ├── station_summary.json         # Consolidated station metrics
│       └── station_summary.md           # Visual markdown station summary report
│
├── global_summary/
│   ├── metadata.json                    # Execution timestamps, platform details, and library versions
│   ├── global_summary.json              # Merged multi-station comparison dataset
│   └── global_summary.md                # Consolidated multi-station markdown diagnostics dashboard
├── pipeline.log                         # Global execution tracer history log
└── errors.log                           # Exceptions and critical warning trace file
```

---

## 🔒 Small Graph Safety Guardrails

- **$N < 10$ features**: Emits warning, continues processing.
- **$N < 5$ features**: Emits warning, still constructs graphs.
- **$N < 3$ features**: Skips the GNN training stage, writes a detailed explanation report to `results/<station>/reports/gnn_explanation.json`, and continues gracefully.
