import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("pipeline")

class DataAuditor:
    def __init__(self, station_name: str, results_dir: str):
        self.station_name = station_name
        self.results_dir = results_dir
        self.audit_dir = os.path.join(results_dir, station_name, "audit")
        os.makedirs(self.audit_dir, exist_ok=True)

    def run_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs a comprehensive data audit on the DataFrame.
        Returns a dict of audit results and saves audit reports to disk.
        """
        logger.info(f"Auditing dataset for station '{self.station_name}'...")
        
        if df.empty:
            logger.warning(f"DataFrame for station '{self.station_name}' is empty. Audit aborted.")
            return {}

        row_count, col_count = df.shape
        mem_usage = df.memory_usage(deep=True).sum()
        mem_usage_mb = mem_usage / (1024 * 1024)
        
        # Missing values
        missing_counts = df.isnull().sum().to_dict()
        missing_pct = (df.isnull().sum() / row_count).to_dict()
        
        # Data types
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Duplicate rows
        duplicate_rows_count = int(df.duplicated().sum())
        
        # Duplicate columns (exact value duplication)
        duplicate_cols = []
        hashes = {}
        for col in df.columns:
            col_hash = pd.util.hash_pandas_object(df[col], index=False).sum()
            if col_hash in hashes:
                if df[col].equals(df[hashes[col_hash]]):
                    duplicate_cols.append((hashes[col_hash], col))
            else:
                hashes[col_hash] = col
                        
        # Unique counts
        unique_counts = df.nunique(dropna=False).to_dict()
        
        # Outlier counts (numeric columns only)
        outliers_iqr = {}
        outliers_zscore = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                outliers_iqr[col] = 0
                outliers_zscore[col] = 0
                continue
                
            # IQR method
            q75, q25 = np.percentile(series, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                outliers_iqr[col] = int(len(iqr_outliers))
            else:
                outliers_iqr[col] = 0
                
            # Z-Score method (threshold = 3)
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_scores = (series - mean) / std
                z_outliers = series[abs(z_scores) > 3]
                outliers_zscore[col] = int(len(z_outliers))
            else:
                outliers_zscore[col] = 0

        audit_results = {
            "station_name": self.station_name,
            "row_count": row_count,
            "column_count": col_count,
            "memory_usage_bytes": int(mem_usage),
            "memory_usage_mb": float(mem_usage_mb),
            "duplicate_rows": duplicate_rows_count,
            "duplicate_columns": duplicate_cols,
            "columns": {}
        }
        
        for col in df.columns:
            audit_results["columns"][col] = {
                "dtype": dtypes[col],
                "missing_count": int(missing_counts[col]),
                "missing_percentage": float(missing_pct[col]),
                "unique_values": int(unique_counts[col]),
                "is_numeric": col in numeric_cols,
                "outliers_iqr": outliers_iqr.get(col, None),
                "outliers_zscore": outliers_zscore.get(col, None)
            }

        # Save to JSON
        json_path = os.path.join(self.audit_dir, "audit_report.json")
        tmp_json_path = json_path + ".tmp"
        with open(tmp_json_path, "w") as f:
            json.dump(audit_results, f, indent=4)
        os.replace(tmp_json_path, json_path)
            
        # Save to Markdown
        md_path = os.path.join(self.audit_dir, "audit_report.md")
        self._save_markdown_report(audit_results, md_path)
        
        logger.info(f"Audit completed. Reports saved to '{self.audit_dir}'.")
        return audit_results

    def _save_markdown_report(self, audit: Dict[str, Any], file_path: str):
        """Generates a readable Markdown summary of the audit."""
        tmp_file_path = file_path + ".tmp"
        with open(tmp_file_path, "w") as f:
            f.write(f"# Data Audit Report: {audit['station_name']}\n\n")
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total Rows:** {audit['row_count']}\n")
            f.write(f"- **Total Columns:** {audit['column_count']}\n")
            f.write(f"- **Memory Usage:** {audit['memory_usage_mb']:.2f} MB\n")
            f.write(f"- **Duplicate Rows:** {audit['duplicate_rows']}\n")
            f.write(f"- **Duplicate Columns (Identical Values):** {len(audit['duplicate_columns'])}\n")
            if audit['duplicate_columns']:
                for pair in audit['duplicate_columns']:
                    f.write(f"  - `{pair[0]}` and `{pair[1]}`\n")
            f.write("\n## Column-level Metrics\n\n")
            f.write("| Column | Data Type | Missing Count | Missing % | Unique Count | IQR Outliers | Z-Score Outliers |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for col, meta in audit['columns'].items():
                iqr = meta['outliers_iqr'] if meta['outliers_iqr'] is not None else "-"
                zsc = meta['outliers_zscore'] if meta['outliers_zscore'] is not None else "-"
                f.write(f"| `{col}` | `{meta['dtype']}` | {meta['missing_count']} | {meta['missing_percentage']*100:.2f}% | {meta['unique_values']} | {iqr} | {zsc} |\n")
        os.replace(tmp_file_path, file_path)
