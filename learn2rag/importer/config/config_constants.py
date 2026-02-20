"""
config_constants.py

Description:
This module contains constants for configuration paths and other global settings.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.7
Creation Date: June 10, 2025
Last Modified: February 20, 2026
"""

# Paths to configuration files
LOGGING_CONFIG_PATH = "config/logging.yaml"
LOGS_DIR = "logs"
JSON_CONFIG_PATH = "config/config.json"
VERSION = "0.0.7"
# Allowed loader types for validation
ALLOWED_LOADERS = ["DirectoryLoader", "CSVLoader", "HTMLLoader", "SharepointLoader"]
