import os
import logging
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("pipeline")

class DataScaler:
    def __init__(self, station_name: str, results_dir: str):
        self.station_name = station_name
        self.results_dir = results_dir
        self.scaled_dir = os.path.join(results_dir, station_name, "scaled")
        os.makedirs(self.scaled_dir, exist_ok=True)

    def scale_data(self, df: pd.DataFrame, classifications: Dict[str, str]) -> pd.DataFrame:
        """
        Applies StandardScaler to all numeric feature columns.
        Leaves timestamps, identifiers, indexes, and targets unscaled.
        Saves scaled.csv.
        """
        logger.info(f"Starting data scaling for station '{self.station_name}'...")
        scaled_df = df.copy()

        # Find columns classified as 'Numeric Feature'
        scale_cols = [
            col for col in df.columns
            if classifications.get(col) == "Numeric Feature" and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not scale_cols:
            logger.warning("No numeric features found to scale. Saving copy of dataset as scaled.csv.")
            scaled_csv_path = os.path.join(self.scaled_dir, "scaled.csv")
            tmp_scaled_csv_path = scaled_csv_path + ".tmp"
            scaled_df.to_csv(tmp_scaled_csv_path, index=False)
            os.replace(tmp_scaled_csv_path, scaled_csv_path)
            return scaled_df

        logger.info(f"Scaling {len(scale_cols)} columns: {scale_cols}")
        
        scaler = StandardScaler()
        scaled_df[scale_cols] = scaler.fit_transform(df[scale_cols])

        scaled_csv_path = os.path.join(self.scaled_dir, "scaled.csv")
        tmp_scaled_csv_path = scaled_csv_path + ".tmp"
        scaled_df.to_csv(tmp_scaled_csv_path, index=False)
        os.replace(tmp_scaled_csv_path, scaled_csv_path)
        logger.info(f"Scaled dataset saved to '{scaled_csv_path}'.")

        return scaled_df
