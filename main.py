import os
import json
import argparse
import traceback
import logging
from typing import List, Dict, Any, Tuple
import multiprocessing
import pandas as pd
import numpy as np

# Import framework modules
from config import Settings
from data_discovery import DataDiscoveryEngine
from audit import DataAuditor
from preprocessing import ColumnClassifier, DataCleaner, DataScaler
from graph_construction import (
    CorrelationGraphBuilder,
    TopKGraphBuilder,
    WeightedGraphBuilder,
    LearnableGraphBuilder
)
from graph_validation import GraphValidator, CentralityAnalyzer
from visualization import Visualizer
from gnn import GNNTrainer, EmbeddingAnalyzer
from reports import ReportGenerator
from utils import setup_global_logging, get_station_logger, set_seed, get_execution_metadata

logger = logging.getLogger("pipeline")

def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, path)

def atomic_write_csv(path: str, df: pd.DataFrame, **kwargs) -> None:
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, **kwargs)
    os.replace(tmp_path, path)

def is_station_success(report: Dict[str, Any]) -> bool:
    if report.get("success") is False:
        return False
    if report.get("status") == "Failed":
        return False
    return True

def process_single_station(args: Tuple[str, List[str], Settings]) -> Dict[str, Any]:
    """
    Worker function to process a single station in isolation.
    Captures all exceptions to prevent pipeline crash.
    """
    station_name, files, settings = args
    
    # Initialize station logger
    station_logger = get_station_logger(station_name, settings.results_dir, settings.log_level)
    station_logger.info(f"==== STARTING PROCESSING FOR STATION: {station_name} ====")
    
    try:
        # 1. Load and merge files
        discovery_engine = DataDiscoveryEngine(settings.data_dir, settings)
        df, file_meta = discovery_engine.load_and_merge_station_files(files)
        
        if df.empty:
            station_logger.error(f"Merged DataFrame is empty for station '{station_name}'. Aborting.")
            return {"station_name": station_name, "success": False, "status": "Failed", "reason": "No data loaded."}

        # 2. Run Data Audit
        auditor = DataAuditor(station_name, settings.results_dir)
        audit_report = auditor.run_audit(df)

        # 3. Column Classification
        classifications = ColumnClassifier.classify_columns(df)

        # 4. Data Cleaning
        cleaner = DataCleaner(station_name, settings.results_dir, settings)
        cleaned_df, cleaning_report = cleaner.clean_data(df, classifications)

        # 5. Scaling
        scaler = DataScaler(station_name, settings.results_dir)
        scaled_df = scaler.scale_data(cleaned_df, classifications)

        # 6. Extract nodes (numeric feature columns remaining in cleaned dataset)
        feature_nodes = [
            col for col in scaled_df.columns
            if classifications.get(col) == "Numeric Feature"
        ]
        
        station_logger.info(f"Identified {len(feature_nodes)} valid features as graph nodes: {feature_nodes}")
        
        if len(feature_nodes) < 2:
            msg = f"Insufficient features ({len(feature_nodes)}) to build graph for station '{station_name}'."
            station_logger.error(msg)
            return {"station_name": station_name, "success": False, "error": msg}

        # Create results/graphs subfolder
        graphs_out_dir = os.path.join(settings.results_dir, station_name, "graphs")
        os.makedirs(graphs_out_dir, exist_ok=True)

        # 7. Construct Graphs
        builders = {
            "correlation": CorrelationGraphBuilder(threshold=settings.correlation_threshold),
            "top_k": TopKGraphBuilder(k=settings.top_k_neighbors),
            "weighted": WeightedGraphBuilder(),
            "learnable": LearnableGraphBuilder(
                epochs=settings.learnable_epochs,
                lr=settings.learnable_lr,
                weight_decay=settings.learnable_weight_decay
            )
        }

        graphs_dict = {}
        graphs_metrics = {}
        visualizer = Visualizer(station_name, settings.results_dir)

        for g_type, builder in builders.items():
            station_logger.info(f"Building {g_type} graph...")
            adj_m, G_nx, edge_list_df = builder.build_graph(scaled_df, feature_nodes)
            
            # Save artifacts
            adj_csv_path = os.path.join(graphs_out_dir, f"{g_type}_adjacency.csv")
            atomic_write_csv(adj_csv_path, pd.DataFrame(adj_m, index=feature_nodes, columns=feature_nodes))
            
            edge_csv_path = os.path.join(graphs_out_dir, f"{g_type}_edgelist.csv")
            atomic_write_csv(edge_csv_path, edge_list_df, index=False)
            
            # Keep in-memory for subsequent steps
            graphs_dict[g_type] = (adj_m, G_nx)

            # Validate Graph
            g_stats = GraphValidator.validate_graph(G_nx, g_type)
            # Add centralities
            centrality_scores = CentralityAnalyzer.compute_centralities(G_nx, g_type)
            g_stats["centralities"] = centrality_scores
            
            graphs_metrics[g_type] = g_stats

            # Visualizations
            is_w = (g_type != "correlation") or settings.weighted_graph
            visualizer.plot_network(G_nx, f"{g_type}_network", f"{g_type.capitalize()} Network Graph", is_weighted=is_w)
            visualizer.plot_heatmap(adj_m, feature_nodes, f"{g_type}_heatmap", f"{g_type.capitalize()} Adjacency Heatmap")
            visualizer.plot_adjacency_grid(adj_m, feature_nodes, f"{g_type}_adj_grid", f"{g_type.capitalize()} Connection Grid")

        # 8. GNN Node Embeddings
        gnn_trainer = GNNTrainer(station_name, settings.results_dir, settings)
        gnn_report = gnn_trainer.run_gnn_pipeline(scaled_df, feature_nodes, graphs_dict)

        # 9. Embedding Quality Analysis & Projection Coordinate Generation
        embed_analyzer = EmbeddingAnalyzer(station_name, settings.results_dir, settings)
        embed_analysis = embed_analyzer.analyze_and_project()

        # Plot Embedding Projections (PCA / t-SNE / UMAP)
        for run_key, analysis in embed_analysis.items():
            nodes = analysis["nodes"]
            projections = analysis.get("projections", {})
            for method, coords in projections.items():
                visualizer.plot_projection(coords, nodes, method, run_key)

        # 10. Generate Station Summary Report
        rep_gen = ReportGenerator(settings.results_dir)
        station_summary = rep_gen.generate_station_report(
            station_name=station_name,
            audit=audit_report,
            cleaning=cleaning_report,
            classifications=classifications,
            graph_stats=graphs_metrics,
            gnn_report=gnn_report,
            embed_report=embed_analysis
        )

        station_logger.info(f"==== COMPLETED PROCESSING FOR STATION: {station_name} ====")
        station_summary["success"] = True
        station_summary["status"] = "Success"
        return station_summary

    except Exception as e:
        tb = traceback.format_exc()
        station_logger.error(f"Unhandled exception during station processing: {e}\n{tb}")
        # Log to global errors trace
        logger.error(f"Error processing station '{station_name}': {e}\n{tb}")
        
        result = {"station_name": station_name, "success": False, "error": str(e), "traceback": tb}
        if settings.strict_mode:
            logger.error("Strict mode enabled. Halting pipeline due to station failure.")
            import sys
            sys.exit(1)
        return result


def main():
    parser = argparse.ArgumentParser(description="Autonomous Graph Analytics Framework (Phase 1)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data folder directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--seed", type=int, default=None, help="Override global seed")
    parser.add_argument("--parallel", type=str, choices=["true", "false"], default=None, help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--log_level", type=str, default=None, help="Global log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--strict", type=str, choices=["true", "false"], default=None, help="Fail the run when any station fails")
    
    args = parser.parse_args()

    # 1. Load configuration and apply arguments override
    settings = Settings.load_from_yaml(args.config)
    
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.results_dir:
        settings.results_dir = args.results_dir
        
    import datetime
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    settings.results_dir = os.path.join(settings.results_dir, f"run_{run_id}")
    if args.seed is not None:
        settings.random_seed = args.seed
    if args.parallel:
        settings.multiprocessing = (args.parallel == "true")
    if args.workers:
        settings.num_workers = args.workers
    if args.log_level:
        settings.log_level = args.log_level
    if args.strict:
        settings.strict_mode = (args.strict == "true")

    settings.validate()

    # Ensure results folder exists
    os.makedirs(settings.results_dir, exist_ok=True)

    # 2. Boot up Logger
    global_logger = setup_global_logging(settings.log_level, settings.results_dir)
    global_logger.info("Initializing Autonomous Graph Analytics Framework...")

    # 3. Initialize reproducibility seed
    set_seed(settings.random_seed)

    # 4. Discover Data
    discovery_engine = DataDiscoveryEngine(settings.data_dir, settings)
    stations_files = discovery_engine.discover_stations()

    if not stations_files:
        global_logger.error("No stations or valid files discovered. Process aborted. Place your datasets inside the data folder.")
        return

    # Build tasks sequence
    tasks = [(station, files, settings) for station, files in stations_files.items()]

    # 5. Process Stations
    all_reports = []
    
    if settings.multiprocessing and len(tasks) > 1:
        num_workers = settings.num_workers or max(1, multiprocessing.cpu_count() - 1)
        global_logger.info(f"Running station tasks in parallel using {num_workers} processes.")
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                all_reports = pool.map(process_single_station, tasks)
        except Exception as e:
            global_logger.error(f"Multiprocessing Pool failed: {e}. Falling back to sequential execution.")
            all_reports = [process_single_station(t) for t in tasks]
    else:
        global_logger.info("Running station tasks sequentially.")
        all_reports = [process_single_station(t) for t in tasks]

    # Filter out failures
    success_reports = [rep for rep in all_reports if is_station_success(rep)]
    failed_reports = [rep for rep in all_reports if not is_station_success(rep)]
    
    global_logger.info(f"Station processing finished. Successes: {len(success_reports)} / Total: {len(all_reports)}.")
    if failed_reports and settings.strict_mode:
        raise SystemExit(f"{len(failed_reports)} station(s) failed in strict mode.")

    # 6. Global Report Consolidation
    if success_reports:
        exec_metadata = get_execution_metadata(settings.random_seed)
        
        # Save raw metadata block
        meta_summary_dir = os.path.join(settings.results_dir, "global_summary")
        os.makedirs(meta_summary_dir, exist_ok=True)
        atomic_write_json(os.path.join(meta_summary_dir, "metadata.json"), exec_metadata)
            
        # Compile global summary md/json
        rep_gen = ReportGenerator(settings.results_dir)
        rep_gen.generate_global_report(success_reports, exec_metadata)
        global_logger.info("Global reports compiled inside 'results/global_summary/'.")
    else:
        global_logger.error("No station tasks succeeded. Skipping global summary generation.")

    global_logger.info("Framework execution completed.")

if __name__ == "__main__":
    main()
