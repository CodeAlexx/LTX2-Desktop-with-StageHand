"""Background thread inference worker.

Runs generation on a daemon thread so DearPyGui stays responsive.
Communicates state back to the UI via a simple shared-state object.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    IDLE = auto()
    RUNNING = auto()
    CANCELLING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class WorkerStatus:
    """Thread-safe status shared between worker and UI."""

    state: WorkerState = WorkerState.IDLE
    phase: str = ""
    progress: float = 0.0
    output_path: str = ""
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "phase": self.phase,
                "progress": self.progress,
                "output_path": self.output_path,
                "error": self.error,
            }


class InferenceWorker:
    """Manages a background thread for video generation."""

    def __init__(self) -> None:
        self.status = WorkerStatus()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    @property
    def is_busy(self) -> bool:
        return self.status.state == WorkerState.RUNNING

    def submit(self, generate_fn: Any, **kwargs: Any) -> None:
        """Start generation on a background thread.

        generate_fn should be pipeline.generate with a progress_cb kwarg.
        """
        if self.is_busy:
            logger.warning("Worker is already busy, ignoring submit")
            return

        self._cancel_event.clear()
        self.status.update(
            state=WorkerState.RUNNING,
            phase="Starting...",
            progress=0.0,
            output_path="",
            error="",
        )

        def _run() -> None:
            try:
                def progress_cb(phase: str, frac: float) -> None:
                    if self._cancel_event.is_set():
                        raise InterruptedError("Cancelled by user")
                    self.status.update(phase=phase, progress=frac)

                result = generate_fn(progress_cb=progress_cb, **kwargs)
                self.status.update(
                    state=WorkerState.DONE,
                    phase="Complete",
                    progress=1.0,
                    output_path=str(result),
                )
            except InterruptedError:
                self.status.update(
                    state=WorkerState.IDLE,
                    phase="Cancelled",
                    progress=0.0,
                )
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Generation failed:\n%s", tb)
                self.status.update(
                    state=WorkerState.ERROR,
                    phase="Error",
                    error=str(e),
                )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the current generation."""
        if self.is_busy:
            self._cancel_event.set()
            self.status.update(state=WorkerState.CANCELLING, phase="Cancelling...")

    def reset(self) -> None:
        """Reset to idle state (call after acknowledging error/done)."""
        self.status.update(
            state=WorkerState.IDLE,
            phase="",
            progress=0.0,
            output_path="",
            error="",
        )
