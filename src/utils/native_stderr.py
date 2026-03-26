r"""Redirect OS-level stderr (fd 2) for native libraries (CUDA, cuQuantum, etc.).

Python's :mod:`warnings` and even :data:`sys.stderr` assignment do not stop C/C++
runtimes from writing to file descriptor 2. Experiment progress uses ``\r`` on
stdout; interleaved native stderr breaks TTY progress lines.

Enable during on-disk experiment solves with::

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
    """Return True when ``HTSP_SILENCE_NATIVE_STDERR`` requests fd 2 redirection.

    Truthy: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Falsy or unset: no redirection.
    """
    raw = os.environ.get("HTSP_SILENCE_NATIVE_STDERR")
    if raw is None or str(raw).strip() == "":
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def resolve_native_stderr_log_path(output_root: Path) -> Path:
    """Resolve log path: ``HTSP_NATIVE_STDERR_LOG`` or *output_root* / ``native_stderr.log``."""
    explicit = os.environ.get("HTSP_NATIVE_STDERR_LOG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (output_root / "native_stderr.log").resolve()


@contextmanager
def redirect_native_stderr_to_file(log_path: Path) -> Generator[None, None, None]:
    """Dup fd 2 to *log_path* for the duration of the context (C + Python stderr).

    Appends to *log_path*. Creates parent directories if needed. Restores fd 2
    and :data:`sys.stderr` on exit.

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
