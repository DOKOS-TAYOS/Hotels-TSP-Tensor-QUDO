"""Tests for OS-level stderr redirection (native libraries)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from utils.native_stderr import redirect_native_stderr_to_file, silence_native_stderr_requested


def test_silence_native_stderr_requested_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTSP_SILENCE_NATIVE_STDERR", raising=False)
    assert silence_native_stderr_requested() is False
    monkeypatch.setenv("HTSP_SILENCE_NATIVE_STDERR", "1")
    assert silence_native_stderr_requested() is True
    monkeypatch.setenv("HTSP_SILENCE_NATIVE_STDERR", "0")
    assert silence_native_stderr_requested() is False


def test_redirect_native_stderr_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "native_err.log"
    with redirect_native_stderr_to_file(log_path):
        print("native-err-test", file=sys.stderr, flush=True)
    text = log_path.read_text(encoding="utf-8")
    assert "native-err-test" in text
    assert sys.stderr is sys.__stderr__
