"""
errors.py

Description:
Error handling for source not available in loaders

Author: Kyrill Meyer
Version: 0.0.1
Institution: IFDT
Creation Date: August 31, 2026
Last Modified: August 31, 2026
"""

class LoaderAccessError(RuntimeError):
    """Raised when a loader cannot access its source or the data source is unusable."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
