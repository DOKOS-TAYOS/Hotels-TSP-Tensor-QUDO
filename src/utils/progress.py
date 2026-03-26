r"""Single-line progress reporter for experiment runs.

Usage (from solvers and workflow):

    from utils.progress import reporter

    # In main workflow, before the instance loop:
    reporter.configure(n_instances=n)

    # Before each solver.solve() call:
    reporter.instance_start(i)

    # After saving results:
    reporter.instance_done(i, str(out_path))

    # Inside optimizer cost_fn or SA loop:
    reporter.opt_step(step, max_steps, energy)

The display always overwrites a single line in-place (\r).
"""

from __future__ import annotations

import os
import sys


class ProgressReporter:
    """Track and print experiment progress on one overwriteable terminal line."""

    def __init__(self) -> None:
        """Create a reporter with TTY detection and zero instance count."""
        self._is_tty: bool = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
        self._n_instances: int = 0
        self._current_instance: int = 0

    def configure(self, n_instances: int) -> None:
        """Set the total number of instances for ``[inst i/n]`` labels.

        Args:
            n_instances: Upper bound shown in progress text.

        """
        self._n_instances = n_instances

    def instance_start(self, i: int) -> None:
        """Print that instance *i* (0-based) is starting.

        Args:
            i: Instance index in the batch.

        """
        # Subprocess pool workers (parallel CUDA-Q / Cirq) must not print; parent owns the TTY line.
        if os.environ.get("HTSP_EXPERIMENT_CUDA_WORKER") == "1":
            return
        self._current_instance = i
        msg = self._fmt_instance(i, "running...")
        self._emit(msg, newline=True)

    def opt_step(self, step: int, max_steps: int, energy: float) -> None:
        r"""Report one optimizer evaluation or simulated-annealing step.

        On a TTY, rewrites one line with ``\r``; otherwise prints sparse
        checkpoints (roughly deciles and the last step).

        Args:
            step: Current step index (1-based display uses internal formatting).
            max_steps: Budget for the bar denominator.
            energy: Current objective value to display.

        """
        if os.environ.get("HTSP_EXPERIMENT_CUDA_WORKER") == "1":
            return
        is_checkpoint = (step % max(1, max_steps // 10) == 0) or (step >= max_steps)

        if not self._is_tty and not is_checkpoint:
            return
        bar = self._bar(step, max_steps)
        msg = (
            f"  {self._fmt_instance(self._current_instance)}"
            f"  step {step:>4}/{max_steps}"
            f"  {bar}"
            f"  E={energy:+.4f}"
        )
        
        if is_checkpoint:
            self._emit(msg, newline=True)
        else:
            self._emit(msg, newline=False)

    def instance_done(self, i: int, path: str) -> None:
        """Print that instance *i* finished and where results were saved.

        Args:
            i: Instance index.
            path: Filesystem path written for this instance.

        """
        if os.environ.get("HTSP_EXPERIMENT_CUDA_WORKER") == "1":
            return
        msg = self._fmt_instance(i, f"saved -> {path}")
        self._emit(msg, newline=True)

    # --- private helpers ---

    def _fmt_instance(self, i: int, suffix: str = "") -> str:
        """Format ``[inst i/n]`` with optional *suffix*."""
        n = self._n_instances if self._n_instances else "?"
        base = f"[inst {i + 1}/{n}]"
        return f"{base}  {suffix}" if suffix else base

    @staticmethod
    def _bar(current: int, total: int, width: int = 20) -> str:
        """Return an ASCII progress bar of fixed *width*."""
        if total == 0:
            return ""
        filled = int(width * current / total)
        return f"[{'#' * filled}{'.' * (width - filled)}]"

    def _emit(self, msg: str, newline: bool = False) -> None:
        """Print *msg*, clearing the line first; optionally end with a newline."""
        if newline:
            print(f"\r\033[K{msg}", end="\n", flush=True)
        else:
            print(f"\r\033[K{msg}", end="", flush=True)


# Module-level singleton — import and use directly from solvers and workflow.
reporter: ProgressReporter = ProgressReporter()
