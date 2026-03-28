"""Redirect OS-level stderr (fd 2) for native libraries (CUDA, cuQuantum, etc.).

Python ``warnings`` and reassigning ``sys.stderr`` do not stop C/C++ runtimes
from writing to file descriptor 2. Experiment progress uses carriage returns
on stdout; interleaved native stderr breaks TTY progress lines.

Enable during on-disk experiment solves::

    HTSP_SILENCE_NATIVE_STDERR=1

Optional explicit log path (default: ``<output_root>/native_stderr.log``)::

    HTSP_NATIVE_STDERR_LOG=/path/to/native_stderr.log
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def silence_native_stderr_requested() -> bool:
    """Return whether environment requests redirecting fd 2 to a log during solves.

    Returns:
        True when ``HTSP_SILENCE_NATIVE_STDERR`` is truthy (``1``, ``true``,
        ``yes``, ``on``, case-insensitive). False when unset or falsy.
    """
    raw = os.environ.get("HTSP_SILENCE_NATIVE_STDERR")
    if raw is None or str(raw).strip() == "":
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def resolve_native_stderr_log_path(output_root: Path) -> Path:
    """Resolve the append path for native stderr redirection.

    Args:
        output_root: Run output root used when no explicit env path is set.

    Returns:
        ``HTSP_NATIVE_STDERR_LOG`` if set, else ``output_root / native_stderr.log``.
    """
    explicit = os.environ.get("HTSP_NATIVE_STDERR_LOG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (output_root / "native_stderr.log").resolve()


@contextmanager
def redirect_native_stderr_to_file(log_path: Path) -> Generator[None, None, None]:
    """Context manager: dup fd 2 to a file (native + Python stderr).

    Appends to ``log_path``, creates parent directories, and restores fd 2 and
    ``sys.stderr`` on exit.

    Args:
        log_path: Append destination for all stderr (native and Python).

    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        sys.stderr.flush()
        os.dup2(log_file.fileno(), stderr_fd)
        sys.stderr = log_file
        yield
    finally:
        sys.stderr.flush()
        log_file.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        log_file.close()
        sys.stderr = sys.__stderr__
