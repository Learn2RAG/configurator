"""
logging_setup.py

Description:
This module provides a function to set up logging configuration for the application.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: June 10, 2025
"""

import logging.config
import yaml

def setup_logging(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)