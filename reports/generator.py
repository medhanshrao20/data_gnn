import os
import json
import logging
import html
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("pipeline")

class ReportGenerator:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.global_dir = os.path.join(results_dir, "global_summary")
        os.makedirs(self.global_dir, exist_ok=True)

    def generate_station_report(self, station_name: str, audit: Dict[str, Any], cleaning: Dict[str, Any],
                                classifications: Dict[str, str], graph_stats: Dict[str, Dict[str, Any]],
                                gnn_report: Dict[str, Any], embed_report: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidates station statistics and writes reports."""
        station_report = {
            "station_name": station_name,
            "metadata": {
                "raw_rows": audit.get("row_count", 0),
                "raw_cols": audit.get("column_count", 0),
                "memory_mb": audit.get("memory_usage_mb", 0.0),
                "duplicate_rows": audit.get("duplicate_rows", 0),
                "duplicate_cols": len(audit.get("duplicate_columns", []))
            },
            "cleaning": cleaning,
            "classifications": classifications,
            "graph_structures": graph_stats,
            "gnn_runs": gnn_report.get("runs", {}),
            "skipped_gnn": gnn_report.get("skipped", False),
            "gnn_skip_reason": gnn_report.get("reason", ""),
            "embeddings_analysis": embed_report
        }

        # Write JSON
        station_rep_dir = os.path.join(self.results_dir, station_name, "reports")
        os.makedirs(station_rep_dir, exist_ok=True)
        json_path = os.path.join(station_rep_dir, "station_summary.json")
        tmp_json_path = json_path + ".tmp"
        with open(tmp_json_path, "w") as f:
            json.dump(station_report, f, indent=4)
        os.replace(tmp_json_path, json_path)

        # Write Markdown
        md_path = os.path.join(station_rep_dir, "station_summary.md")
        self._write_station_markdown(station_report, md_path)
        logger.info(f"Generated summary reports for '{station_name}' at '{station_rep_dir}'.")
        
        return station_report

    def generate_global_report(self, all_station_reports: List[Dict[str, Any]], exec_metadata: Dict[str, Any]):
        """Consolidates all stations and graph types into a global diagnostic report."""
        global_data = {
            "execution_metadata": exec_metadata,
            "stations": all_station_reports
        }
        
        # Write JSON
        json_path = os.path.join(self.global_dir, "global_summary.json")
        tmp_json_path = json_path + ".tmp"
        with open(tmp_json_path, "w") as f:
            json.dump(global_data, f, indent=4)
        os.replace(tmp_json_path, json_path)

        # Write Markdown
        md_path = os.path.join(self.global_dir, "global_summary.md")
        self._write_global_markdown(global_data, md_path)
        logger.info(f"Generated global summary report at '{md_path}'.")

    def _write_station_markdown(self, rep: Dict[str, Any], path: str):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding='utf-8') as f:
            def md_escape(text):
                # Basic markdown escaping for tables
                text = str(text)
                text = text.replace('|', '\\|')
                text = text.replace('\n', ' ')
                text = text.replace('`', '\\`')
                return html.escape(text)

            safe_station = md_escape(rep['station_name'])
            f.write(f"# Station Analytics Report: {safe_station}\n\n")
            
            f.write("## 1. Raw Dataset Profile\n\n")
            f.write(f"- **Raw Rows:** {rep['metadata']['raw_rows']}\n")
            f.write(f"- **Raw Columns:** {rep['metadata']['raw_cols']}\n")
            f.write(f"- **Memory Footprint:** {rep['metadata']['memory_mb']:.2f} MB\n")
            f.write(f"- **Duplicate Rows Detected:** {rep['metadata']['duplicate_rows']}\n")
            f.write(f"- **Identical Columns Detected:** {rep['metadata']['duplicate_cols']}\n\n")

            f.write("## 2. Preprocessing & Column Classification\n\n")
            f.write("### Schema Classification Results\n\n")
            f.write("| Column Name | Classification Type |\n")
            f.write("| --- | --- |\n")
            for col, cls in rep['classifications'].items():
                safe_col = md_escape(col)
                f.write(f"| `{safe_col}` | **{cls}** |\n")
            f.write("\n")

            f.write("### Cleaning Operations\n\n")
            f.write(f"- **Imputation Strategy Chosen:** `{rep['cleaning'].get('imputation_strategy', 'None')}`\n")
            f.write(f"- **Duplicate Rows Dropped:** {rep['cleaning'].get('dropped_rows', 0)}\n")
            
            dropped_cols = rep['cleaning'].get('dropped_columns', {})
            if dropped_cols:
                f.write("- **Dropped Columns:**\n")
                for c, r in dropped_cols.items():
                    safe_col = md_escape(c)
                    f.write(f"  - `{safe_col}`: {r}\n")
            else:
                f.write("- **Dropped Columns:** None\n")

            redundant_cols = rep['cleaning'].get('redundant_columns_detected', {})
            if redundant_cols:
                f.write("- **Redundant Columns Detected (Not Dropped):**\n")
                for c, r in redundant_cols.items():
                    safe_col = md_escape(c)
                    f.write(f"  - `{safe_col}`: {r}\n")
            else:
                f.write("- **Redundant Columns Detected:** None\n")
                
            outliers = rep['cleaning'].get('outliers_treated', {})
            if outliers:
                f.write("- **Outliers Clipped/Winsorized:**\n")
                for c, details in outliers.items():
                    safe_col = md_escape(c)
                    f.write(f"  - `{safe_col}`: {details['count']} outliers treated (bounds: [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}])\n")
            else:
                f.write("- **Outliers Clipped:** None\n")
            f.write("\n")

            f.write("## 3. Graph Construction Diagnostic Metrics\n\n")
            f.write("| Metric | Correlation Graph | Top-K Graph | Weighted Graph | Learnable Graph |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            g_types = ["correlation", "top_k", "weighted", "learnable"]
            metrics = [
                ("Node Count", "node_count"),
                ("Edge Count", "edge_count"),
                ("Density", "density"),
                ("Average Degree", "avg_degree"),
                ("Connected Components", "connected_components"),
                ("Isolated Nodes", "isolated_nodes"),
                ("Clustering Coefficient", "clustering_coefficient"),
                ("Diameter (CC)", "diameter"),
                ("Radius (CC)", "radius")
            ]
            
            for label, key in metrics:
                row_str = f"| {label} "
                for gt in g_types:
                    gt_stats = rep['graph_structures'].get(gt, {})
                    val = gt_stats.get(key, "-")
                    if isinstance(val, float):
                        row_str += f"| {val:.4f} "
                    else:
                        row_str += f"| {val} "
                row_str += "|\n"
                f.write(row_str)
            f.write("\n")

            f.write("## 4. GNN Embeddings & Validation\n\n")
            if rep['skipped_gnn']:
                f.write(f"> [!WARNING]\n> GNN learning was skipped for this station. Reason: {rep.get('gnn_skip_reason') or 'Too few features.'}\n\n")
            else:
                f.write("### GNN Training Loss Summary (Autoencoder Reconstruction)\n\n")
                f.write("| Configuration | Graph Type | GNN Architecture | Final Loss | Status |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                for run_key, run_val in rep['gnn_runs'].items():
                    fl = run_val['final_loss']
                    loss_str = f"{fl:.6f}" if isinstance(fl, (int, float)) else str(fl)
                    f.write(f"| `{run_key}` | {run_val['graph_type']} | {run_val['model_name']} | {loss_str} | {run_val['validation']} |\n")
                f.write("\n")
                
                f.write("### Embedding Quality Diagnostics\n\n")
                f.write("| Configuration | Pairwise Distance (Mean) | Pairwise Distance (Std) | Average Variance | Sparsity |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                for run_key, analysis in rep['embeddings_analysis'].items():
                    metrics = analysis.get("metrics", {})
                    f.write(f"| `{run_key}` | {metrics.get('avg_pairwise_distance', 0.0):.4f} | {metrics.get('std_pairwise_distance', 0.0):.4f} | {metrics.get('dimension_variance', 0.0):.4f} | {metrics.get('sparsity', 0.0)*100:.2f}% |\n")
                f.write("\n")
        os.replace(tmp_path, path)

    def _write_global_markdown(self, glob: Dict[str, Any], path: str):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding='utf-8') as f:
            f.write("# Global Consolidated Analytics Summary\n\n")
            
            f.write("## 1. System Metadata & Environment\n\n")
            meta = glob['execution_metadata']
            f.write(f"- **Execution Timestamp:** {meta.get('timestamp', 'N/A')}\n")
            f.write(f"- **Framework Random Seed:** {meta.get('seed', 'N/A')}\n")
            f.write(f"- **OS/Platform:** {meta.get('platform', 'N/A')}\n")
            f.write(f"- **Python Version:** {meta.get('python_version', 'N/A')}\n")
            f.write(f"- **PyTorch Available:** {meta.get('pytorch_available', 'N/A')} (GPU: {meta.get('cuda_available', 'N/A')})\n\n")

            f.write("## 2. Station Profiles Comparison\n\n")
            f.write("| Station Name | Raw Rows | Raw Columns | Clean Columns (Nodes) | Memory (MB) | GNN Status |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for st in glob['stations']:
                st_meta = st['metadata']
                gnn_stat = "Skipped" if st['skipped_gnn'] else "Success"
                # Find number of nodes
                sample_gt = list(st['graph_structures'].values())
                node_count = sample_gt[0].get('node_count', 0) if sample_gt else 0
                safe_station = html.escape(str(st['station_name']))
                f.write(f"| `{safe_station}` | {st_meta['raw_rows']} | {st_meta['raw_cols']} | {node_count} | {st_meta['memory_mb']:.2f} | {gnn_stat} |\n")
            f.write("\n")

            f.write("## 3. Graph Strategies Density Comparison\n\n")
            f.write("| Station Name | Correlation Density | Top-K Density | Weighted Density | Learnable Density |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for st in glob['stations']:
                s_structs = st['graph_structures']
                c_d = s_structs.get("correlation", {}).get("density", 0.0)
                t_d = s_structs.get("top_k", {}).get("density", 0.0)
                w_d = s_structs.get("weighted", {}).get("density", 0.0)
                l_d = s_structs.get("learnable", {}).get("density", 0.0)
                safe_station = html.escape(str(st['station_name']))
                f.write(f"| `{safe_station}` | {c_d:.4f} | {t_d:.4f} | {w_d:.4f} | {l_d:.4f} |\n")
            f.write("\n")

            f.write("## 4. Embedding Quality Summary (Cross-Model Average Pairwise Distance)\n\n")
            f.write("| Station Name | GAT Mean Dist | GraphSAGE Mean Dist |\n")
            f.write("| --- | --- | --- |\n")
            for st in glob['stations']:
                safe_station = html.escape(str(st['station_name']))
                if st['skipped_gnn']:
                    f.write(f"| `{safe_station}` | Skipped | Skipped |\n")
                    continue
                
                # average across graph types
                gat_dists = []
                sage_dists = []
                for run_key, analysis in st['embeddings_analysis'].items():
                    avg_dist = analysis.get("metrics", {}).get("avg_pairwise_distance", 0.0)
                    if "GAT" in run_key:
                        gat_dists.append(avg_dist)
                    else:
                        sage_dists.append(avg_dist)
                
                mean_gat = np.mean(gat_dists) if gat_dists else 0.0
                mean_sage = np.mean(sage_dists) if sage_dists else 0.0
                f.write(f"| `{safe_station}` | {mean_gat:.4f} | {mean_sage:.4f} |\n")
            f.write("\n")
            
            f.write("> [!NOTE]\n> Detailed visualizations, edge lists, and raw node embedding matrices can be found in their respective station subfolders under `results/`.\n")
