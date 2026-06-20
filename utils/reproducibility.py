import os
import random
import sys
import platform
import logging
import json
from datetime import datetime
import numpy as np
import torch
from typing import Dict, Any

logger = logging.getLogger("pipeline")

def set_seed(seed: int = 42):
    """Sets standard seeds for random number generators to ensure reproducible runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}.")

def get_execution_metadata(seed: int) -> Dict[str, Any]:
    """Gathers runtime environment configuration and package versions."""
    package_versions = {}
    packages = [
        "pandas", "numpy", "scikit-learn", "scipy", "torch", 
        "matplotlib", "seaborn", "networkx", "pyyaml", "umap-learn"
    ]
    
    for pkg in packages:
        # Map import name to package name if different
        import_name = "sklearn" if pkg == "scikit-learn" else pkg
        import_name = "umap" if pkg == "umap-learn" else import_name
        
        try:
            mod = __import__(import_name)
            package_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            package_versions[pkg] = "not installed"

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_available": "yes" if "torch" in sys.modules else "no",
        "cuda_available": "yes" if ("torch" in sys.modules and torch.cuda.is_available()) else "no",
        "device_count": torch.cuda.device_count() if ("torch" in sys.modules and torch.cuda.is_available()) else 0,
        "device_name": torch.cuda.get_device_name(0) if ("torch" in sys.modules and torch.cuda.is_available()) else "CPU",
        "packages": package_versions
    }
    
    return metadata
