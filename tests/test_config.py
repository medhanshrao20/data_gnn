import pytest
from config.settings import Settings

def test_settings_validation():
    # Valid settings should pass
    s = Settings()
    s.validate()
    
    # Invalid ranges should raise ValueError
    with pytest.raises(ValueError, match="correlation_threshold must be between"):
        s = Settings()
        s.correlation_threshold = 1.5
        s.validate()
        
    with pytest.raises(ValueError, match="correlation_threshold must be between"):
        s = Settings()
        s.correlation_threshold = -0.5
        s.validate()

    with pytest.raises(ValueError, match="top_k_neighbors must be positive"):
        s = Settings()
        s.top_k_neighbors = 0
        s.validate()

    with pytest.raises(ValueError, match="gnn_epochs must be positive"):
        s = Settings()
        s.gnn_epochs = -10
        s.validate()

    with pytest.raises(ValueError, match="gnn_lr must be positive"):
        s = Settings()
        s.gnn_lr = 0.0
        s.validate()
        
    with pytest.raises(ValueError, match="redundant_correlation_threshold must be between"):
        s = Settings()
        s.redundant_correlation_threshold = 1.1
        s.validate()
        
    with pytest.raises(ValueError, match="missing_threshold_drop must be between"):
        s = Settings()
        s.missing_threshold_drop = -0.1
        s.validate()
