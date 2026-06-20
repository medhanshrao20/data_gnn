import numpy as np
import pandas as pd
import pytest
import torch
from gnn.models import GATModel, GraphSAGEModel
from gnn.trainer import GNNTrainer
from config import Settings

def test_gnn_layers_forward():
    # Setup dummy features (N=5 nodes, D=10 dimension) and dummy symmetric adjacency (5x5)
    N = 5
    D = 10
    h = torch.randn(N, D)
    adj = torch.tensor([
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0]
    ])
    
    # 1. Test GAT Forward Pass
    gat = GATModel(in_features=D, hidden_dim=8, out_features=4)
    gat.eval()
    with torch.no_grad():
        out_gat = gat(h, adj)
    assert out_gat.shape == (N, 4)
    assert not torch.isnan(out_gat).any()

    # 2. Test GraphSAGE Forward Pass
    sage = GraphSAGEModel(in_features=D, hidden_dim=8, out_features=4, aggregator_type="mean")
    sage.eval()
    with torch.no_grad():
        out_sage = sage(h, adj)
    assert out_sage.shape == (N, 4)
    assert not torch.isnan(out_sage).any()

def test_gat_uses_signed_edge_values():
    torch.manual_seed(42)
    h = torch.randn(3, 4)
    adj_positive = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    adj_negative = torch.tensor([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0]
    ])

    gat = GATModel(in_features=4, hidden_dim=5, out_features=2, dropout=0.0)
    gat.eval()
    with torch.no_grad():
        out_positive = gat(h, adj_positive)
        out_negative = gat(h, adj_negative)

    assert not torch.allclose(out_positive, out_negative)

def test_gnn_pipeline_and_safety_checks(tmp_path):
    settings = Settings()
    settings.gnn_epochs = 5
    settings.results_dir = str(tmp_path / "test_results")
    
    trainer = GNNTrainer("test_gnn_station", settings.results_dir, settings)
    
    # CASE 1: Tiny graph safety check (N < 3 features) -> GNN should be skipped
    df_tiny = pd.DataFrame({
        "time": ["2026-06-01", "2026-06-02"],
        "feat1": [1.0, 2.0],
        "feat2": [10.0, 20.0]
    })
    feature_cols_tiny = ["feat1", "feat2"]
    graphs_dict_tiny = {"correlation": (np.zeros((2,2)), None)}
    
    report_tiny = trainer.run_gnn_pipeline(df_tiny, feature_cols_tiny, graphs_dict_tiny)
    assert report_tiny["skipped"] is True
    assert "minimum 3 required" in report_tiny["reason"]

    # CASE 2: Valid graph (N = 4 features) -> GNN should run successfully
    df_valid = pd.DataFrame({
        "time": ["2026-06-01", "2026-06-02", "2026-06-03"],
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [2.0, 1.0, 3.0],
        "feat3": [3.0, 2.0, 1.0],
        "feat4": [1.0, 3.0, 2.0]
    })
    feature_cols_valid = ["feat1", "feat2", "feat3", "feat4"]
    adj_mock = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
    ])
    graphs_dict_valid = {"correlation": (adj_mock, None)}
    
    report_valid = trainer.run_gnn_pipeline(df_valid, feature_cols_valid, graphs_dict_valid)
    assert report_valid["skipped"] is False
    assert "correlation_GAT" in report_valid["runs"]
    assert "correlation_GraphSAGE" in report_valid["runs"]
