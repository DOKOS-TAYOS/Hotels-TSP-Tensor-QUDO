"""Tests for CUDA-Q target selection."""

from __future__ import annotations

import pytest


def test_ensure_cudaq_target_selects_nvidia_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected GPU must configure the NVIDIA backend."""
    pytest.importorskip("cudaq")
    from solvers.cudaq_solver import cudaq_target

    set_target_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
    monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: name == "nvidia")

    def _fake_set_target(name: str, option: str | None = None) -> None:
        set_target_calls.append((name, option))

    monkeypatch.setattr(cudaq_target.cudaq, "set_target", _fake_set_target)

    cudaq_target.ensure_cudaq_target()

    assert set_target_calls == [("nvidia", "fp64")]


def test_ensure_cudaq_target_raises_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No compatible GPU must raise a clear error instead of using CPU fallback."""
    pytest.importorskip("cudaq")
    from solvers.cudaq_solver import cudaq_target

    monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 0)
    monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

    def _unexpected_set_target(*args, **kwargs) -> None:
        raise AssertionError("set_target should not be called without a detected GPU")

    monkeypatch.setattr(cudaq_target.cudaq, "set_target", _unexpected_set_target)

    with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
        cudaq_target.ensure_cudaq_target()


def test_ensure_cudaq_target_raises_without_nvidia_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA-Q install without NVIDIA target support must fail clearly."""
    pytest.importorskip("cudaq")
    from solvers.cudaq_solver import cudaq_target

    monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
    monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: False)

    def _unexpected_set_target(*args, **kwargs) -> None:
        raise AssertionError("set_target should not be called without the NVIDIA target")

    monkeypatch.setattr(cudaq_target.cudaq, "set_target", _unexpected_set_target)

    with pytest.raises(RuntimeError, match="NVIDIA target is unavailable"):
        cudaq_target.ensure_cudaq_target()
