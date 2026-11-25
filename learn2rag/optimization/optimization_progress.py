import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class OptimizationProgress:
    total_trials: int
    status_file: Path
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def emit(self, message: str) -> None:
        with self.status_file.open("a", encoding="utf-8") as f:
            f.write(f"{message} {datetime.now():%Y-%m-%d %H:%M}\n")

    def start(self) -> None:
        self.emit("Optimization started")

    def trial_started(self, trial: int) -> None:
        elapsed = self.elapsed_seconds()
        completed = trial - 1

        eta = (
            _format_duration(elapsed / completed * (self.total_trials - completed))
            if completed else "calculating"
        )

        self.emit(
            f"Trial {trial}/{self.total_trials} started | "
            f"elapsed {_format_duration(elapsed)} | "
            f"estimated remaining time {eta}"
        )

    def finish(self) -> None:
        self.emit(
            f"Optimization finished | "
            f"total elapsed time: {_format_duration(self.elapsed_seconds())}"
        )