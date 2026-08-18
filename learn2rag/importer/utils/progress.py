"""
progress.py

Description:
This module provides progress tracking for the import process.

Author: Kyrill Meyer
Institution: IFDT
Version: 0.0.1
Creation Date: June 29, 2026
Last Modified: June 29, 2026
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional


statusLogger = logging.getLogger("status")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(processed: int, total: int, elapsed_seconds: float) -> Optional[str]:
    if processed <= 0 or total <= 0 or processed >= total:
        return None
    remaining = (elapsed_seconds / processed) * (total - processed)
    return _format_duration(remaining)


@dataclass
class ImportProgress:
    total_loaders: int
    mode: str = "full"
    started_at: float = field(default_factory=time.perf_counter)
    current_loader_index: Optional[int] = None
    current_loader_type: Optional[str] = None
    current_source: Optional[str] = None

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def emit(
        self,
        phase: str,
        message: str,
        *,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        loader_index: Optional[int] = None,
        loader_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        active_loader_index = loader_index if loader_index is not None else self.current_loader_index
        active_loader_type = loader_type if loader_type is not None else self.current_loader_type
        active_source = source if source is not None else self.current_source

        parts = [phase]
        if active_loader_index is not None and self.total_loaders > 0:
            parts.append(f"loader {active_loader_index}/{self.total_loaders}")
        if active_loader_type:
            parts.append(active_loader_type)
        if active_source:
            parts.append(active_source)
        parts.append(message)

        elapsed = self.elapsed_seconds()
        if processed is not None:
            progress_text = f"processed {processed}"
            if total is not None and total > 0:
                percentage = (processed / total) * 100
                progress_text = f"processed {processed}/{total} ({percentage:.1f}%)"
                eta = _format_eta(processed, total, elapsed)
                if eta:
                    progress_text += f" | eta {eta}"
            parts.append(progress_text)

        parts.append(f"elapsed {_format_duration(elapsed)}")
        statusLogger.info(" | ".join(parts))

    def start_import(self) -> None:
        self.emit("Phase 1/4 Init", f"Import started | mode {self.mode} | loaders {self.total_loaders}")

    def start_loader(self, loader_index: int, loader_type: str, source: str) -> None:
        self.current_loader_index = loader_index
        self.current_loader_type = loader_type
        self.current_source = source
        self.emit("Phase 2/4 Load", "Loader started")

    def finish_loader(self, document_count: int) -> None:
        self.emit("Phase 2/4 Load", f"Loader finished | documents {document_count}")

    def start_indexing(self, document_count: int) -> None:
        self.emit("Phase 3/4 Index", f"Indexing started | documents {document_count}")

    def finish_import(self, total_documents: Optional[int] = None) -> None:
        details = "Import finished"
        if total_documents is not None:
            details += f" | documents {total_documents}"
        self.emit("Phase 4/4 Done", details)

    def fail_import(self, reason: str = "Import failed") -> None:
        self.emit("Phase 4/4 Done", reason)