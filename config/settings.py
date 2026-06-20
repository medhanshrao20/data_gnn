import os
import yaml
from typing import Any, Dict

class Settings:
    def __init__(self, config_dict: Dict[str, Any] = None):
        # Default Configurations
        self.data_dir = "data"
        self.results_dir = "results"
        self.random_seed = 42
        self.timestamp_col = None
        
        # Execution
        self.strict_mode = False
        self.drop_duplicates = True
        self.drop_redundant_columns = True
        
        # Preprocessing Configurations
        self.outlier_method = "iqr"  # 'iqr', 'z_score', 'isolation_forest'
        self.outlier_treatment = "clip"  # 'clip', 'winsorize'
        self.missing_impute_strategy = "auto"  # 'auto', 'interpolate', 'median', 'knn'
        self.missing_threshold_drop = 0.5  # drop columns with > 50% missing values
        self.near_constant_threshold = 1e-4  # variance threshold for constant cols
        self.redundant_correlation_threshold = 0.99  # drop highly correlated feature columns
        
        # Graph Construction Configurations
        self.correlation_threshold = 0.5  # threshold for correlation graph edge
        self.top_k_neighbors = 5  # default K for top-k graph
        self.weighted_graph = True
        self.learnable_epochs = 150
        self.learnable_lr = 0.01
        self.learnable_weight_decay = 1e-4
        
        # GNN Configurations
        self.gnn_epochs = 200
        self.gnn_lr = 0.01
        self.gnn_hidden_dim = 16
        self.gnn_embedding_dim = 8
        self.gnn_weight_decay = 1e-4
        
        # Runtime Configurations
        self.multiprocessing = True
        self.num_workers = None  # defaults to os.cpu_count()
        self.log_level = "INFO"
        
        # Overwrite with dict if provided
        if config_dict:
            self.update(config_dict)

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Settings":
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    inst = cls()
                    for k in data.keys():
                        if not hasattr(inst, k):
                            raise ValueError(f"Unknown configuration key: {k}")
                    inst.update(data)
                    inst.validate()
                    return inst
        inst = cls()
        inst.validate()
        return inst

    def validate(self):
        valid_impute = {"auto", "interpolate", "median", "knn"}
        valid_outlier_methods = {"iqr", "z_score", "isolation_forest"}
        valid_outlier_treatments = {"clip", "winsorize", "none"}
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.timestamp_col is not None and not isinstance(self.timestamp_col, str):
            raise ValueError("timestamp_col must be a string when set")
        if not isinstance(self.strict_mode, bool):
            raise ValueError("strict_mode must be true or false")
        if not isinstance(self.drop_duplicates, bool):
            raise ValueError("drop_duplicates must be true or false")
        if not isinstance(self.drop_redundant_columns, bool):
            raise ValueError("drop_redundant_columns must be true or false")
        if self.missing_impute_strategy not in valid_impute:
            raise ValueError(f"missing_impute_strategy must be one of {sorted(valid_impute)}")
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"outlier_method must be one of {sorted(valid_outlier_methods)}")
        if self.outlier_treatment not in valid_outlier_treatments:
            raise ValueError(f"outlier_treatment must be one of {sorted(valid_outlier_treatments)}")
        if self.correlation_threshold < 0.0 or self.correlation_threshold > 1.0:
            raise ValueError("correlation_threshold must be between 0.0 and 1.0")
        if self.top_k_neighbors <= 0:
            raise ValueError("top_k_neighbors must be positive")
        if self.learnable_epochs <= 0:
            raise ValueError("learnable_epochs must be positive")
        if self.learnable_lr <= 0:
            raise ValueError("learnable_lr must be positive")
        if self.learnable_weight_decay < 0:
            raise ValueError("learnable_weight_decay must be non-negative")
        if self.gnn_epochs <= 0:
            raise ValueError("gnn_epochs must be positive")
        if self.gnn_lr <= 0:
            raise ValueError("gnn_lr must be positive")
        if self.gnn_hidden_dim <= 0:
            raise ValueError("gnn_hidden_dim must be positive")
        if self.gnn_embedding_dim <= 0:
            raise ValueError("gnn_embedding_dim must be positive")
        if self.gnn_weight_decay < 0:
            raise ValueError("gnn_weight_decay must be non-negative")
        if self.redundant_correlation_threshold < 0.0 or self.redundant_correlation_threshold > 1.0:
            raise ValueError("redundant_correlation_threshold must be between 0.0 and 1.0")
        if self.missing_threshold_drop < 0.0 or self.missing_threshold_drop > 1.0:
            raise ValueError("missing_threshold_drop must be between 0.0 and 1.0")
        if self.near_constant_threshold < 0.0:
            raise ValueError("near_constant_threshold must be non-negative")
        if self.num_workers is not None and self.num_workers <= 0:
            raise ValueError("num_workers must be positive when set")
        if str(self.log_level).upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {sorted(valid_log_levels)}")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
