"""
logging_setup.py

Description:
This module provides a function to set up logging configuration for the application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: June 10, 2025
Last Modified: February 20, 2026
"""

import importlib.resources
import yaml
import logging.config
from typing import Optional, Union, Any
from pathlib import Path

def setup_logging(config_path: Optional[Union[str, Path, Any]] = None) -> None:
    """
    Set up logging configuration from a YAML file.
    If config_path is provided, use it; otherwise, fall back to default.
    """
    if config_path is None:
        # Fallback: Paket-relativer Pfad
        config_path = importlib.resources.files("learn2rag.importer.config") / "logging.yaml"
    
    try:
        with open(str(config_path), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Logging configuration file not found at {config_path}")
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Error loading logging configuration: {e}")