"""Shared test helpers and fixtures."""

from pathlib import Path
import shutil
from uuid import uuid4


def workspace_tmp_dir(prefix: str) -> Path:
    """Create a temporary directory under tests/.tmp for isolated test runs."""
    base_dir = Path(__file__).resolve().parent / ".tmp"
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def cleanup_workspace_tmp_dir(temp_dir: Path) -> None:
    """Remove a temporary directory and its parent if empty."""
    shutil.rmtree(temp_dir, ignore_errors=True)
    base_dir = temp_dir.parent
    if base_dir.exists() and not any(base_dir.iterdir()):
        base_dir.rmdir()
