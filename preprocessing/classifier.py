import logging
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("pipeline")

class ColumnClassifier:
    @staticmethod
    def classify_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Classifies each column in the DataFrame.
        Possible classifications:
        - Timestamp
        - Identifier
        - Sequential Index
        - Numeric Feature
        - Categorical Feature
        - Target Candidate
        - Unknown
        """
        classifications = {}
        row_count = len(df)
        
        if row_count == 0:
            return classifications

        for col in df.columns:
            series = df[col]
            col_lower = col.lower()
            
            # 1. Target Candidate detection by name
            target_keywords = ['target', 'label', 'outcome', 'y_val', 'class_label']
            if any(kw == col_lower or col_lower.endswith(f'_{kw}') for kw in target_keywords):
                classifications[col] = "Target Candidate"
                continue
                
            # 2. Datetime/Timestamp Detection
            # If dtype is already datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                classifications[col] = "Timestamp"
                continue
                
            # Try parsing if it is object/string or integer representation of time
            if series.dtype == object or pd.api.types.is_integer_dtype(series):
                # Sample a subset to speed up datetime inference
                sample = series.dropna().head(100)
                if not sample.empty:
                    try:
                        time_names = ['time', 'date', 'timestamp', 'epoch', 'datetime', 'ts']
                        is_time_name = any(tn in col_lower for tn in time_names)
                        
                        # Only parse integers if they have large values (like epoch ts) or explicit time names
                        if pd.api.types.is_integer_dtype(series):
                            if not (is_time_name or sample.max() > 1e9):
                                # Skip parsing small integers as dates
                                raise ValueError("Small integers are not timestamps")

                        # Try parsing sample
                        parsed = pd.to_datetime(sample, errors='coerce')
                        # If > 80% parses successfully and there is variance in dates
                        if (parsed.notna().sum() / len(sample)) > 0.8 and parsed.nunique() > 1:
                            classifications[col] = "Timestamp"
                            continue
                    except Exception:
                        pass

            # 3. Identifier & Sequential Index
            # Sequential Index: strictly monotonic, integer type, starts close to 0 or 1
            if pd.api.types.is_integer_dtype(series):
                if series.is_monotonic_increasing:
                    diffs = series.diff().dropna()
                    if (diffs == 1).all() or col_lower in ['index', 'idx', 'row_num', 'rownum']:
                        classifications[col] = "Sequential Index"
                        continue

            # Identifier: high uniqueness, integer or string, names containing id/key/uuid
            id_keywords = ['id', 'key', 'uuid', 'guid', 'station', 'code', 'pk', 'fk']
            is_id_name = any(col_lower == kw or col_lower.endswith(f'_{kw}') or col_lower.startswith(f'{kw}_') for kw in id_keywords)
            unique_ratio = series.nunique() / row_count
            
            if is_id_name and (unique_ratio > 0.8 or series.dtype == object):
                classifications[col] = "Identifier"
                continue
            
            if series.dtype == object and unique_ratio == 1.0:
                classifications[col] = "Identifier"
                continue

            # 4. Categorical Feature
            # Object, category, boolean, or low-cardinality integers
            if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == bool or series.dtype == object:
                classifications[col] = "Categorical Feature"
                continue
                
            if pd.api.types.is_integer_dtype(series):
                # If unique counts are very low (e.g. <= 20 or < 5% of row counts)
                if series.nunique() <= 20 or (unique_ratio < 0.05 and series.nunique() < 100):
                    classifications[col] = "Categorical Feature"
                    continue

            # 5. Numeric Feature
            if pd.api.types.is_numeric_dtype(series):
                classifications[col] = "Numeric Feature"
                continue
                
            # 6. Unknown
            classifications[col] = "Unknown"

        # Double check: if no target column is classified, but one column has a strong target name
        # We can scan the classification.
        logger.info(f"Column classification complete. Details: {classifications}")
        return classifications
