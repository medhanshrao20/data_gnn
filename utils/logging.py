import os
import logging
import sys
from typing import Any

def setup_global_logging(log_level: str = "INFO", results_dir: str = "results") -> logging.Logger:
    """
    Sets up the global logging environment.
    Creates:
      - Console logging (stdout)
      - pipeline.log (general execution history)
      - errors.log (errors and exceptions only)
    """
    os.makedirs(results_dir, exist_ok=True)
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Root-level pipeline logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)
    
    # Global pipeline.log handler
    pipeline_log_path = os.path.join(results_dir, "pipeline.log")
    pipeline_handler = logging.FileHandler(pipeline_log_path, encoding='utf-8')
    pipeline_handler.setFormatter(formatter)
    pipeline_handler.setLevel(numeric_level)
    logger.addHandler(pipeline_handler)
    
    # Global errors.log handler
    errors_log_path = os.path.join(results_dir, "errors.log")
    error_handler = logging.FileHandler(errors_log_path, encoding='utf-8')
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    logger.info("Global logging configured successfully.")
    return logger


def get_station_logger(station_name: str, results_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Creates a logger specific to a station.
    Saves to results/<station>/logs/station.log.
    """
    import re
    safe_station = re.sub(r'[^A-Za-z0-9_\-]', '_', station_name)
    station_log_dir = os.path.join(results_dir, safe_station)
    os.makedirs(station_log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"pipeline.station.{station_name}")
    log_file = os.path.join(station_log_dir, f"{safe_station}.log")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent propagation to avoid double printing to stdout
    logger.propagate = False
    
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (to keep showing stdout for station actions)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Station specific file logger
    station_log_path = os.path.join(station_log_dir, "station.log")
    file_handler = logging.FileHandler(station_log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Station logger initialized for '{station_name}'.")
    return logger
