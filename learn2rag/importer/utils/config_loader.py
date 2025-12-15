"""
config_loader.py

Description:
This module provides a function to load configuration files (e.g., JSON, YAML) for the application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: June 10, 2025
"""

import json
import yaml

def load_json_config(config_path):
    """Load a JSON configuration file with error handling."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON configuration file: {config_path}")
 
def validate_config_entry(entry):
    """Validate individual configuration entries based on their type."""
    loader_type = entry.get("loader_type")
    path = entry.get("path")
    recursive = entry.get("recursive")
    url = entry.get("url")

    if not loader_type:
        raise ValueError("Missing 'loader_type' in configuration entry.")
    
    if loader_type == "DirectoryLoader":
        if not path:
            raise ValueError("Missing 'path' for 'DirectoryLoader' in configuration entry.")
        if not recursive:
            raise ValueError("Missing 'recursive' flag for 'DirectoryLoader'. Please set it to True or False.")
    elif loader_type == "CSVLoader":
        if not path:
            raise ValueError("Missing 'path' for 'CSVLoader' in configuration entry.")
    elif loader_type == "HTMLLoader":
        if not url:
            raise ValueError("Missing 'url' to access for 'HTMLLoader' in configuration entry.")  
    else:
        raise ValueError(f"Unknown 'loader_type': {loader_type}")

    # ToDo: Add further validation for other loader types as needed
    return True