"""
config_loader.py

Description:
This module provides a function to load configuration files (e.g., JSON, YAML) for the application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.3
Creation Date: June 10, 2025
Last Modified: February 20, 2026
"""

import json
import yaml
from typing import Dict, Any, cast
from ..config.config_constants import ALLOWED_LOADERS

def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load a JSON configuration file with error handling."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            return cast(Dict[str, Any], config)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON configuration file: {config_path}")
 
def validate_config_entry(entry: Dict[str, Any]) -> bool:
    """Validate individual configuration entries based on their type."""
    loader_type = entry.get("loader_type")
    if not loader_type:
        raise ValueError("Missing 'loader_type' in configuration entry.")
    
    if loader_type not in ALLOWED_LOADERS:
        raise ValueError(f"Unknown 'loader_type': {loader_type}")

    
    if loader_type == "DirectoryLoader":
        if not entry.get("path"):
            raise ValueError("Missing 'path' for 'DirectoryLoader' in configuration entry.")
        if not entry.get("recursive"):
            raise ValueError("Missing 'recursive' flag for 'DirectoryLoader'. Please set it to True or False.")
    elif loader_type == "CSVLoader":
        if not entry.get("path"):
            raise ValueError("Missing 'path' for 'CSVLoader' in configuration entry.")
    elif loader_type == "HTMLLoader":
        if not entry.get("url"):
            raise ValueError("Missing 'url' to access for 'HTMLLoader' in configuration entry.") 
    elif loader_type == "SharepointLoader":  
        if not entry.get("client_id") or not entry.get("client_secret") or not entry.get("tenant_id") or not entry.get("document_library_id"):
            raise ValueError("Missing required parameters for SharePointLoader: client_id, client_secret, tenant_id, document_library_id.")
     
    else:
        raise ValueError(f"Unknown 'loader_type': {loader_type}")

    # ToDo: Add further validation for other loader types as needed
    return True