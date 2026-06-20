import os
import tempfile
import pandas as pd
import pytest
from data_discovery.discovery import DataDiscoveryEngine

def test_station_and_file_discovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy station directories
        station_a = os.path.join(tmpdir, "Station_A")
        station_b = os.path.join(tmpdir, "Station_B")
        os.makedirs(station_a)
        os.makedirs(station_b)

        # Create dummy files
        df_a1 = pd.DataFrame({"timestamp": ["2026-06-01", "2026-06-02"], "value": [10.0, 12.0]})
        df_a2 = pd.DataFrame({"timestamp": ["2026-06-03"], "value": [15.0]})
        df_b1 = pd.DataFrame({"timestamp": ["2026-06-01"], "temp": [25.0]})

        df_a1.to_csv(os.path.join(station_a, "data_1.csv"), index=False)
        df_a2.to_csv(os.path.join(station_a, "data_2.csv"), index=False)
        df_b1.to_parquet(os.path.join(station_b, "data_1.parquet"), index=False)

        engine = DataDiscoveryEngine(tmpdir)
        stations = engine.discover_stations()

        assert "Station_A" in stations
        assert "Station_B" in stations
        assert len(stations["Station_A"]) == 2
        assert len(stations["Station_B"]) == 1

def test_load_and_merge_schema_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create station and schema-drifted files
        station_path = os.path.join(tmpdir, "TestStation")
        os.makedirs(station_path)
        
        file1 = os.path.join(station_path, "f1.csv")
        file2 = os.path.join(station_path, "f2.csv")
        
        df1 = pd.DataFrame({"time": ["2026-06-01"], "val_a": [1.0]})
        # df2 adds a new column 'val_b' and drops 'val_a'
        df2 = pd.DataFrame({"time": ["2026-06-02"], "val_b": [2.0]})
        
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        engine = DataDiscoveryEngine(tmpdir)
        with pytest.raises(ValueError, match="Schema drift detected"):
            engine.load_and_merge_station_files([file1, file2])
