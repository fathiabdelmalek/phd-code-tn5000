import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class Logger:
    """Simple logger for training metrics and text output."""

    def __init__(
        self,
        exp_dir: str,
        name: str = "train",
        console: bool = True,
        file: bool = True,
    ):
        self.exp_dir = Path(exp_dir)
        self.logs_dir = self.exp_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.console = console
        self.file = file

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"{name}_{timestamp}.log"

        self._metrics_history: List[Dict] = []

    def info(self, message: str) -> None:
        """Log an info message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] INFO: {message}"

        if self.console:
            print(formatted)

        if self.file:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")

    def warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] WARNING: {message}"

        if self.console:
            print(formatted, file=sys.stderr)

        if self.file:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")

    def error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] ERROR: {message}"

        if self.console:
            print(formatted, file=sys.stderr)

        if self.file:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics dictionary."""
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        self._metrics_history.append(entry)

        step_str = f"Step {step}" if step is not None else "Metrics"
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        self.info(f"{step_str}: {metrics_str}")

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar value."""
        self.log_metrics({name: value}, step)

    def log_text(self, text: str) -> None:
        """Log a text message."""
        self.info(text)

    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """Save all logged metrics to a JSON file."""
        if filepath is None:
            filepath = self.logs_dir / "metrics.json"

        with open(filepath, "w") as f:
            json.dump(self._metrics_history, f, indent=2, default=str)

    def get_history(self) -> List[Dict]:
        """Return the metrics history."""
        return self._metrics_history


def get_logger(exp_dir: str, name: str = "train") -> Logger:
    """Factory function to create a logger."""
    return Logger(exp_dir=exp_dir, name=name)
