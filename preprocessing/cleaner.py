import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("pipeline")

class DataCleaner:
    def __init__(self, station_name: str, results_dir: str, settings: Any):
        self.station_name = station_name
        self.results_dir = results_dir
        self.settings = settings
        self.clean_dir = os.path.join(results_dir, station_name, "cleaned")
        self.reports_dir = os.path.join(results_dir, station_name, "reports")
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def clean_data(self, df: pd.DataFrame, classifications: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans the DataFrame:
        1. Drop duplicate rows.
        2. Standardize invalid values (inf, -inf -> NaN).
        3. Drop columns with excessive missingness (> threshold).
        4. Detect and drop constant / near-constant columns.
        5. Detect and drop redundant columns (correlation > 0.99).
        6. Impute missing values (Interpolate, Median, or KNN).
        7. Detect and treat outliers (IQR, Z-Score, Isolation Forest).
        Returns:
            Cleaned DataFrame and execution report.
        """
        logger.info(f"Starting data cleaning for station '{self.station_name}'...")
        
        # Copy to avoid setting with copy warnings
        cleaned_df = df.copy()
        report = {
            "dropped_columns": {},
            "redundant_columns_detected": {},
            "outliers_treated": {},
            "imputation_strategy": ""
        }

        # 0. Handle duplicate rows explicitly based on policy
        initial_rows = len(cleaned_df)
        if getattr(self.settings, "drop_duplicates", True):
            cleaned_df = cleaned_df.drop_duplicates()
            dropped_rows = initial_rows - len(cleaned_df)
            if dropped_rows > 0:
                logger.info(f"Removed {dropped_rows} duplicate rows.")
            report["dropped_rows"] = dropped_rows
        else:
            report["dropped_rows"] = 0

        # 1. ENUM validations
        valid_impute = ["auto", "interpolate", "median", "knn"]
        if self.settings.missing_impute_strategy not in valid_impute:
            raise ValueError(f"Invalid missing_impute_strategy: {self.settings.missing_impute_strategy}")
            
        valid_outlier_methods = ["iqr", "z_score", "isolation_forest"]
        if self.settings.outlier_method not in valid_outlier_methods:
            raise ValueError(f"Invalid outlier_method: {self.settings.outlier_method}")
            
        valid_outlier_treatments = ["clip", "winsorize", "none"]
        if self.settings.outlier_treatment not in valid_outlier_treatments:
            raise ValueError(f"Invalid outlier_treatment: {self.settings.outlier_treatment}")

        # 2. Handle invalid values (inf, -inf -> NaN)
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                inf_mask = np.isinf(cleaned_df[col])
                inf_count = int(inf_mask.sum())
                if inf_count > 0:
                    logger.warning(f"Column '{col}' has {inf_count} infinite values. Replacing with NaN.")
                    cleaned_df.loc[inf_mask, col] = np.nan

        # 3. Drop columns with excessive missing values
        cols_to_drop = []
        for col in cleaned_df.columns:
            # Skip checking Timestamp or Index if classified
            cls = classifications.get(col, "Unknown")
            if cls in ["Timestamp", "Identifier", "Sequential Index"]:
                continue
                
            null_pct = cleaned_df[col].isnull().sum() / len(cleaned_df)
            if null_pct > self.settings.missing_threshold_drop:
                cols_to_drop.append(col)
                report["dropped_columns"][col] = f"Excessive missingness ({null_pct*100:.1f}%)"
                
        if cols_to_drop:
            logger.info(f"Dropping columns due to excessive missingness: {cols_to_drop}")
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)

        # 4. Remove constant and near-constant columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        const_cols = []
        for col in numeric_cols:
            cls = classifications.get(col, "Unknown")
            if cls in ["Timestamp", "Identifier", "Sequential Index"]:
                continue
                
            var = cleaned_df[col].var(ddof=0)
            if pd.isna(var) or var <= self.settings.near_constant_threshold:
                const_cols.append(col)
                report["dropped_columns"][col] = f"Constant / Near-constant (variance = {var})"
                
        if const_cols:
            logger.info(f"Dropping constant/near-constant columns: {const_cols}")
            cleaned_df = cleaned_df.drop(columns=const_cols)
            numeric_cols = [c for c in numeric_cols if c not in const_cols]

        # 5. Missing value imputation
        # Identify columns requiring imputation (excluding indexes, target, timestamps)
        impute_cols = []
        for col in cleaned_df.columns:
            cls = classifications.get(col, "Unknown")
            if cls in ["Numeric Feature", "Categorical Feature", "Unknown"] and cleaned_df[col].isnull().any():
                impute_cols.append(col)

        if impute_cols:
            # Determine imputation strategy
            strategy = self.settings.missing_impute_strategy
            if strategy == "auto":
                # If timestamp is present, we assume ordering exists and try interpolation first
                has_timestamp = any(classifications.get(c) == "Timestamp" for c in cleaned_df.columns)
                if has_timestamp and len(cleaned_df) > 5:
                    strategy_to_apply = "interpolate"
                elif len(cleaned_df) <= 10000:
                    strategy_to_apply = "knn"
                else:
                    strategy_to_apply = "median"
            elif strategy == "knn":
                if len(cleaned_df) > 10000:
                    logger.warning("KNN imputation requested but row count > 10000. Falling back to median to prevent OOM/stall.")
                    strategy_to_apply = "median"
                else:
                    strategy_to_apply = "knn"
            else:
                strategy_to_apply = strategy
            
            report["imputation_strategy"] = strategy_to_apply
            logger.info(f"Imputing missing values using '{strategy_to_apply}' strategy for columns: {impute_cols}")
            
            if strategy_to_apply == "interpolate":
                has_timestamp = any(classifications.get(c) == "Timestamp" for c in cleaned_df.columns)
                if not has_timestamp:
                    raise ValueError("Interpolation requires a timestamp column to ensure sorted, regular time index.")
                timestamp_col = [c for c in cleaned_df.columns if classifications.get(c) == "Timestamp"][0]
                parsed_ts = pd.to_datetime(cleaned_df[timestamp_col], errors="coerce", utc=True)
                if parsed_ts.isna().any():
                    raise ValueError(f"Interpolation requires valid timestamps in '{timestamp_col}'.")
                cleaned_df = cleaned_df.assign(_parsed_timestamp=parsed_ts).sort_values("_parsed_timestamp")
                if not cleaned_df["_parsed_timestamp"].is_monotonic_increasing:
                    raise ValueError(f"Timestamp column '{timestamp_col}' must be monotonic after sorting.")
                # Interpolate and then backfill/forward fill any remaining NaNs
                for col in impute_cols:
                    if col in numeric_cols:
                        time_indexed = cleaned_df.set_index("_parsed_timestamp")[col]
                        cleaned_df[col] = time_indexed.interpolate(method='time').ffill().bfill().to_numpy()
                    else:
                        # Categorical or non-numeric: use mode
                        cleaned_df[col] = cleaned_df[col].ffill().bfill()
                cleaned_df = cleaned_df.drop(columns=["_parsed_timestamp"])
                        
            elif strategy_to_apply == "knn" and len(numeric_cols) > 0:
                # Impute numeric columns with KNNImputer
                knn_cols = [c for c in impute_cols if c in numeric_cols]
                if knn_cols:
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        cleaned_df[knn_cols] = imputer.fit_transform(cleaned_df[knn_cols])
                    except Exception as e:
                        logger.error(f"KNN Imputer failed: {e}. Falling back to median imputation.")
                        strategy_to_apply = "median"
                
                # Fill non-numeric with mode
                non_num_cols = [c for c in impute_cols if c not in numeric_cols]
                for col in non_num_cols:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "missing")
                    
            if strategy_to_apply == "median" or (strategy_to_apply == "knn" and not numeric_cols):
                for col in impute_cols:
                    if col in numeric_cols:
                        median_val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                    else:
                        mode_val = cleaned_df[col].mode()
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0] if not mode_val.empty else "missing")

        # 6. Remove redundant columns (correlation > configured threshold)
        redundant_cols = []
        if len(numeric_cols) > 1:
            corr_matrix = cleaned_df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            for col in upper_tri.columns:
                # Find if any other column has correlation > 0.99 with this one
                high_corr = upper_tri[col][upper_tri[col] > self.settings.redundant_correlation_threshold]
                if not high_corr.empty:
                    redundant_cols.append(col)
                    correlated_with = list(high_corr.index)
                    report["redundant_columns_detected"][col] = f"Highly correlated with {correlated_with}"
                    
        if redundant_cols:
            if getattr(self.settings, "drop_redundant_columns", True):
                logger.info(f"Dropping highly correlated redundant columns: {redundant_cols}")
                cleaned_df = cleaned_df.drop(columns=redundant_cols)
                numeric_cols = [c for c in numeric_cols if c not in redundant_cols]
                for col in redundant_cols:
                    report["dropped_columns"][col] = report["redundant_columns_detected"][col]
                report["redundant_columns_detected"] = {}
            else:
                logger.info(f"Identified highly correlated redundant columns (not dropping): {redundant_cols}")

        # 7. Outlier treatment (numeric features only)
        # We only treat features that are numeric features (not index, timestamp, target)
        outlier_features = [c for c in numeric_cols if classifications.get(c) == "Numeric Feature"]
        if outlier_features:
            method = self.settings.outlier_method
            treatment = self.settings.outlier_treatment
            logger.info(f"Detecting and treating outliers using method '{method}' and treatment '{treatment}'...")
            
            if method == "iqr":
                for col in outlier_features:
                    series = cleaned_df[col]
                    q75, q25 = np.percentile(series, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outlier_mask = (series < lower_bound) | (series > upper_bound)
                        outliers_count = int(outlier_mask.sum())
                        if outliers_count > 0:
                            report["outliers_treated"][col] = {
                                "count": outliers_count,
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound)
                            }
                            if treatment == "clip":
                                cleaned_df[col] = np.clip(series, lower_bound, upper_bound)
                            elif treatment == "winsorize":
                                p1, p99 = np.percentile(series, [1, 99])
                                cleaned_df[col] = np.clip(series, p1, p99)
                                
            elif method == "z_score":
                for col in outlier_features:
                    series = cleaned_df[col]
                    mean = series.mean()
                    std = series.std()
                    if std > 0:
                        z_scores = (series - mean) / std
                        outlier_mask = abs(z_scores) > 3
                        outliers_count = int(outlier_mask.sum())
                        if outliers_count > 0:
                            lower_bound = float(mean - 3 * std)
                            upper_bound = float(mean + 3 * std)
                            report["outliers_treated"][col] = {
                                "count": outliers_count,
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound
                            }
                            if treatment == "clip":
                                cleaned_df[col] = np.clip(series, lower_bound, upper_bound)
                            elif treatment == "winsorize":
                                p1, p99 = np.percentile(series, [1, 99])
                                cleaned_df[col] = np.clip(series, p1, p99)
                                
            elif method == "isolation_forest":
                # Fit Isolation Forest across all numeric features at once
                try:
                    iso = IsolationForest(random_state=self.settings.random_seed, n_estimators=100)
                    # Fit on imputed numeric features
                    preds = iso.fit_predict(cleaned_df[outlier_features])
                    outlier_mask = preds == -1
                    total_outliers = int(outlier_mask.sum())
                    if total_outliers > 0:
                        logger.info(f"Isolation Forest identified {total_outliers} outlier rows.")
                        report["isolation_forest_outliers"] = total_outliers
                        if treatment == "clip" or treatment == "winsorize":
                            for col in outlier_features:
                                p1, p99 = np.percentile(cleaned_df[col], [1, 99])
                                cleaned_df.loc[outlier_mask, col] = np.clip(cleaned_df.loc[outlier_mask, col], p1, p99)
                except Exception as e:
                    logger.error(f"Isolation Forest outlier detection failed: {e}. Skipping outlier treatment.")

        # Save cleaned data
        cleaned_csv_path = os.path.join(self.clean_dir, "cleaned.csv")
        tmp_cleaned_csv_path = cleaned_csv_path + ".tmp"
        cleaned_df.to_csv(tmp_cleaned_csv_path, index=False)
        os.replace(tmp_cleaned_csv_path, cleaned_csv_path)
        logger.info(f"Cleaned dataset saved to '{cleaned_csv_path}'.")

        # Save report
        report_json_path = os.path.join(self.reports_dir, "cleaning_report.json")
        tmp_report_json_path = report_json_path + ".tmp"
        with open(tmp_report_json_path, "w") as f:
            json.dump(report, f, indent=4)
        os.replace(tmp_report_json_path, report_json_path)

        return cleaned_df, report
