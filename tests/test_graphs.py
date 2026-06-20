import numpy as np
import pandas as pd
import pytest
from graph_construction.correlation import CorrelationGraphBuilder
from graph_construction.top_k import TopKGraphBuilder
from graph_construction.weighted import WeightedGraphBuilder
from graph_construction.learnable import LearnableGraphBuilder

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    # Generate 3 correlated columns and 1 uncorrelated
    x1 = np.random.randn(100)
    x2 = x1 * 0.9 + np.random.randn(100) * 0.1
    x3 = x1 * -0.8 + np.random.randn(100) * 0.2
    x4 = np.random.randn(100)
    
    df = pd.DataFrame({
        "feat1": x1, "feat2": x2, "feat3": x3, "feat4": x4
    })
    return df, ["feat1", "feat2", "feat3", "feat4"]

def test_correlation_graph(dummy_data):
    df, features = dummy_data
    # High threshold to only connect highly correlated (1-2 and 1-3)
    builder = CorrelationGraphBuilder(threshold=0.7)
    adj, G, edges = builder.build_graph(df, features)
    
    assert adj.shape == (4, 4)
    # Check node labels match feature names
    assert list(G.nodes) == features
    # Check that feat1 is connected to feat2 and feat3, but not feat4
    assert G.has_edge("feat1", "feat2")
    assert G.has_edge("feat1", "feat3")
    assert not G.has_edge("feat1", "feat4")

def test_top_k_graph(dummy_data):
    df, features = dummy_data
    builder = TopKGraphBuilder(k=2)
    adj, G, edges = builder.build_graph(df, features)
    
    assert adj.shape == (4, 4)
    # Each node must have out-degree exactly K
    assert all(G.out_degree(node) == 2 for node in features)

def test_weighted_graph(dummy_data):
    df, features = dummy_data
    builder = WeightedGraphBuilder()
    adj, G, edges = builder.build_graph(df, features)
    
    assert adj.shape == (4, 4)
    # Graph is fully connected, so N*(N-1)/2 = 6 undirected edges
    assert G.number_of_edges() == 6
    # No self-loops: diagonal is zero
    assert np.allclose(np.diag(adj), 0.0)

def test_learnable_graph(dummy_data):
    df, features = dummy_data
    builder = LearnableGraphBuilder(epochs=10, lr=0.1)
    adj, G, edges = builder.build_graph(df, features)
    
    assert adj.shape == (4, 4)
    # Check symmetry
    assert np.allclose(adj, adj.T, atol=1e-5)
    # Check zero-diagonal (no self loops)
    assert np.allclose(np.diag(adj), 0.0)
    # Weights should be normalized in [-1, 1]
    assert np.all(adj >= -1.0) and np.all(adj <= 1.0)
