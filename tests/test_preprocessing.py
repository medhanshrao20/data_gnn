import os
import pandas as pd
import numpy as np
import pytest
from preprocessing.classifier import ColumnClassifier
from preprocessing.cleaner import DataCleaner
from preprocessing.scaler import DataScaler
from config import Settings

def test_column_classifier():
    df = pd.DataFrame({
        "timestamp_col": ["2026-06-01", "2026-06-02", "2026-06-03"],
        "index": [0, 1, 2],
        "feature_1": [1.2, 3.4, 5.6],
        "feature_category": ["type_a", "type_b", "type_a"],
        "station_id": ["ST_1", "ST_1", "ST_1"],
        "y_target": [100.0, 105.0, 110.0]
    })
    
    classes = ColumnClassifier.classify_columns(df)
    
    assert classes["timestamp_col"] == "Timestamp"
    assert classes["index"] == "Sequential Index"
    assert classes["feature_1"] == "Numeric Feature"
    assert classes["feature_category"] == "Categorical Feature"
    assert classes["station_id"] == "Identifier"
    assert classes["y_target"] == "Target Candidate"

def test_cleaner_impute_and_drop():
    # Setup dataset with:
    # - a missing value
    # - a constant column
    # - duplicate rows
    # - redundant correlated columns (feature_a and feature_b identical)
    df = pd.DataFrame({
        "time": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-03"],
        "feat_constant": [5.0, 5.0, 5.0, 5.0],
        "feat_missing": [1.0, np.nan, 3.0, 3.0],
        "feat_a": [10.0, 15.0, 30.0, 30.0],
        "feat_b": [10.0, 15.0, 30.0, 30.0],
    })
    
    classifications = {
        "time": "Timestamp",
        "feat_constant": "Numeric Feature",
        "feat_missing": "Numeric Feature",
        "feat_a": "Numeric Feature",
        "feat_b": "Numeric Feature"
    }

    settings = Settings()
    settings.missing_impute_strategy = "median"
    settings.near_constant_threshold = 1e-5
    
    cleaner = DataCleaner("test_clean", "test_results", settings)
    cleaned_df, report = cleaner.clean_data(df, classifications)
    
    # 1. Row deduplication: duplicates dropped
    assert len(cleaned_df) == 3
    
    # 2. Imputation check (median of 1.0, 3.0 -> 2.0)
    assert cleaned_df["feat_missing"].isna().sum() == 0
    assert cleaned_df["feat_missing"].iloc[1] == 2.0
    
    # 3. Constant dropping
    assert "feat_constant" not in cleaned_df.columns
    
    # 4. Redundant correlation dropping (feat_a and feat_b identical)
    assert "feat_b" not in cleaned_df.columns
    assert "feat_a" in cleaned_df.columns
    assert "feat_b" in report["dropped_columns"]

    # Cleanup test output folder
    if os.path.exists("test_results"):
        import shutil
        shutil.rmtree("test_results")

def test_scaler():
    df = pd.DataFrame({
        "time": ["2026-06-01", "2026-06-02", "2026-06-03"],
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [10.0, 20.0, 30.0],
        "target": [0.0, 1.0, 0.0]
    })
    classifications = {
        "time": "Timestamp",
        "feature_1": "Numeric Feature",
        "feature_2": "Numeric Feature",
        "target": "Target Candidate"
    }
    
    scaler = DataScaler("test_scale", "test_results")
    scaled_df = scaler.scale_data(df, classifications)
    
    # Scaling checked: mean of scaled feature_1 must be ~0.0
    assert np.allclose(scaled_df["feature_1"].mean(), 0.0, atol=1e-5)
    # Target and time must remain unscaled
    assert scaled_df["target"].equals(df["target"])
    assert scaled_df["time"].equals(df["time"])

    # Cleanup test output folder
    import os
    if os.path.exists("test_results"):
        import shutil
        shutil.rmtree("test_results")
