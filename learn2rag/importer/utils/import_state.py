"""
import_state.py

Description:
    Manages the persistent per-loader import state (last_import_timestamp).
    The state file (import_state.json) is stored next to the importer_config.json.

    State file format:
        {
            "loader_id": {
                "last_import_timestamp": "2026-04-27T10:00:00+00:00"
            }
        }

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: April 27, 2026
Last Modified: April 27, 2026
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Learn2RAGImporter")


class ImportState:
    """
    Manages import state (timestamps) per loader.

    Tracks the start timestamp of the last successful import run per loader_id.
    Changes are held in memory until ``save_success()`` is called, so a failed
    import does not advance the stored timestamp.
    """

    def __init__(self, state_file_path: str) -> None:
        """
        Initialise ImportState and load existing state from disk if present.

        Args:
            state_file_path (str): Absolute or relative path to the JSON state file
                                   (e.g. ``/data/import_state.json``).
        """
        self._path = Path(state_file_path)
        self._state: Dict[str, Any] = {}
        self._pending: Dict[str, datetime] = {}  # in-memory only, not yet persisted
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    self._state = json.load(f)
                logger.info("Import state loaded from %s", self._path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load import state from %s: %s — starting fresh.", self._path, e)
                self._state = {}
        else:
            self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)
        logger.debug("Import state saved to %s", self._path)

    def get_last_import_time(self, loader_id: str) -> Optional[datetime]:
        """
        Return the start timestamp of the last successful import for a loader.

        Args:
            loader_id (str): Unique loader identifier as defined in the importer config.

        Returns:
            Optional[datetime]: UTC-aware datetime of the last successful import start,
                                or ``None`` if no state exists for this loader.
        """
        entry = self._state.get(loader_id)
        if not entry:
            return None
        ts_str = entry.get("last_import_timestamp")
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str)
            # If no timezone info is present, treat as UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError as e:
            logger.warning("Invalid timestamp for loader_id=%s: %s", loader_id, e)
            return None

    def record_import_start(self, loader_id: str, timestamp: datetime) -> None:
        """
        Record the start time of an ongoing import in memory (not yet persisted).

        Must be called before ``save_success()`` for the same loader_id.

        Args:
            loader_id (str): Unique loader identifier.
            timestamp (datetime): UTC-aware datetime representing the import start time.
        """
        self._pending[loader_id] = timestamp

    def save_success(self, loader_id: str) -> None:
        """
        Persist the previously recorded import start timestamp to disk.

        Must be preceded by a call to ``record_import_start()`` for the same
        loader_id; raises ``AssertionError`` otherwise.

        Args:
            loader_id (str): Unique loader identifier.
        """
        assert loader_id in self._pending, (
            f"save_success() called for loader_id='{loader_id}' without prior record_import_start()"
        )
        ts = self._pending.pop(loader_id)
        self._state[loader_id] = {
            "last_import_timestamp": ts.isoformat()
        }
        self._save()
        logger.info("Import state saved for loader_id=%s (start_time=%s)", loader_id, ts.isoformat())
