"""Single-line progress reporter for experiment runs.

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

import sys


class ProgressReporter:
    """Tracks and displays experiment progress on a single rewritable line."""

    def __init__(self) -> None:
        self._is_tty: bool = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
        self._n_instances: int = 0
        self._current_instance: int = 0

    def configure(self, n_instances: int) -> None:
        """Set total instance count."""
        self._n_instances = n_instances

    def instance_start(self, i: int) -> None:
        """Report that instance *i* (0-based) is about to be solved."""
        self._current_instance = i
        msg = self._fmt_instance(i, "running...")
        self._emit(msg, newline=True)

    def opt_step(self, step: int, max_steps: int, energy: float) -> None:
        """Report one optimizer evaluation or SA iteration.

        Always rewrites the same line (\r). Never commits with \n.
        """
        is_checkpoint = (step % (max_steps//10) == 0) or (step >= max_steps)

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
        """Report that instance *i* has been saved."""
        msg = self._fmt_instance(i, f"saved -> {path}")
        self._emit(msg, newline=True)

    # --- private helpers ---

    def _fmt_instance(self, i: int, suffix: str = "") -> str:
        n = self._n_instances if self._n_instances else "?"
        base = f"[inst {i + 1}/{n}]"
        return f"{base}  {suffix}" if suffix else base

    @staticmethod
    def _bar(current: int, total: int, width: int = 20) -> str:
        if total == 0:
            return ""
        filled = int(width * current / total)
        return f"[{'#' * filled}{'.' * (width - filled)}]"

    def _emit(self, msg: str, newline: bool = False) -> None:
        if newline:
            print(f"\r\033[K{msg}", end="\n", flush=True)
        else:
            print(f"\r\033[K{msg}", end="", flush=True)


# Module-level singleton — import and use directly from solvers and workflow.
reporter: ProgressReporter = ProgressReporter()
