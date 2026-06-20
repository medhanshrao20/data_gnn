import os
import glob
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("pipeline")

class DataDiscoveryEngine:
    def __init__(self, data_dir: str, settings: Any = None):
        self.data_dir = data_dir
        self.settings = settings

    def discover_stations(self) -> Dict[str, List[str]]:
        """
        Discovers stations and their associated data files in data_dir.
        If there are no subdirectories but files exist in data_dir, they are grouped under 'root_station'.
        """
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory '{self.data_dir}' does not exist. Creating it.")
            os.makedirs(self.data_dir, exist_ok=True)
            return {}

        stations = {}
        # Get all subdirectories
        entries = os.listdir(self.data_dir)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(self.data_dir, e))]
        
        # Supported file patterns
        extensions = ['*.csv', '*.parquet', '*.pq', '*.xlsx', '*.xls', '*.tsv']

        if len(subdirs) > 0:
            for subdir in subdirs:
                subdir_path = os.path.join(self.data_dir, subdir)
                files = []
                for ext in extensions:
                    # Recursive search
                    files.extend(glob.glob(os.path.join(subdir_path, '**', ext), recursive=True))
                if files:
                    stations[subdir] = sorted(list(set(files)))
        else:
            # Check if there are data files directly in the root
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            if files:
                stations['root_station'] = sorted(list(set(files)))

        logger.info(f"Discovered {len(stations)} station(s): {list(stations.keys())}")
        for station, files in stations.items():
            logger.info(f"Station '{station}' has {len(files)} files.")
        
        return stations

    def load_and_merge_station_files(self, files: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads and merges multiple files for a single station.
        Handles schema drift and sorts chronologically if timestamp exists.
        Returns:
            Merged DataFrame, and file metadata dictionary.
        """
        dfs = []
        file_inventory = []
        schema_drift_detected = False
        all_columns = set()
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            _, ext = os.path.splitext(file_path.lower())
            
            try:
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                elif ext in ['.parquet', '.pq']:
                    df = pd.read_parquet(file_path)
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif ext == '.tsv':
                    df = pd.read_csv(file_path, sep='\t')
                else:
                    logger.warning(f"Unsupported file format: {file_path}. Skipping.")
                    continue
                
                if df.empty:
                    logger.warning(f"File {file_path} is empty. Skipping.")
                    continue
                
                if len(all_columns) > 0 and set(df.columns) != all_columns:
                    schema_drift_detected = True
                    raise ValueError(f"Schema drift detected in {file_name}. Columns do not match preceding files.")
                
                all_columns.update(df.columns)
                
                file_inventory.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "rows": df.shape[0],
                    "cols": df.shape[1],
                    "columns": list(df.columns)
                })
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}. Failing station.")
                raise
        
        if not dfs:
            return pd.DataFrame(), {"file_inventory": [], "schema_drift": False, "total_rows": 0, "total_cols": 0}

        # Merge all dataframes. Concat aligns columns and fills missing values with NaN.
        merged_df = pd.concat(dfs, ignore_index=True, sort=False)
        
        # Sort chronologically if a timestamp is present
        timestamp_col = getattr(self.settings, "timestamp_col", None)
        if not timestamp_col or timestamp_col not in merged_df.columns:
            timestamp_col = self._find_likely_timestamp_column(merged_df)
            
        if timestamp_col:
            # Parse dates unambiguously to UTC and check for invalid rows
            temp_ts = pd.to_datetime(merged_df[timestamp_col], errors='coerce', utc=True)
            if temp_ts.isna().any():
                logger.error(f"Invalid dates found in timestamp column '{timestamp_col}'. Failing station.")
                raise ValueError(f"Invalid dates in timestamp column '{timestamp_col}'.")
            
            merged_df = merged_df.iloc[temp_ts.argsort()]
            logger.info(f"Sorted merged data chronologically by likely timestamp column '{timestamp_col}'.")

        metadata = {
            "file_inventory": file_inventory,
            "schema_drift": schema_drift_detected,
            "total_rows": merged_df.shape[0],
            "total_cols": merged_df.shape[1]
        }
        
        return merged_df, metadata

    def _find_likely_timestamp_column(self, df: pd.DataFrame) -> str:
        """Find a timestamp column by name and parse quality, avoiding accidental matches."""
        exact_names = ["timestamp", "datetime", "date_time", "time", "date", "epoch", "ts"]
        candidates = []
        for col in df.columns:
            col_lower = col.lower()
            score = 0
            if col_lower in exact_names:
                score = 3
            elif any(token in col_lower.split("_") for token in exact_names):
                score = 2
            elif any(ind in col_lower for ind in ["timestamp", "datetime", "date", "time"]):
                score = 1
            if score == 0:
                continue

            parsed = pd.to_datetime(df[col].dropna().head(200), errors="coerce", utc=True)
            if len(parsed) == 0:
                continue
            parse_ratio = parsed.notna().sum() / len(parsed)
            if parse_ratio >= 0.8 and parsed.nunique() > 1:
                candidates.append((score, parse_ratio, col))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]
        return ""
