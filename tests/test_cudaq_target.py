"""Tests for CUDA-Q target selection."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_and_import(monkeypatch: pytest.MonkeyPatch):
    """Import cudaq_target and reset its module-level cache."""
    pytest.importorskip("cudaq")
    from solvers.cudaq_solver import cudaq_target

    cudaq_target.reset_target_state()
    return cudaq_target


# ---------------------------------------------------------------------------
# Noiseless target selection
# ---------------------------------------------------------------------------


class TestNoiselessTarget:
    """Noiseless (``noise_config=None``) target selection."""

    def test_selects_nvidia_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A detected GPU must configure the NVIDIA backend."""
        cudaq_target = _reset_and_import(monkeypatch)

        set_target_calls: list[tuple[str, str | None]] = []

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: name == "nvidia")

        def _fake_set_target(name: str, option: str | None = None) -> None:
            set_target_calls.append((name, option))

        monkeypatch.setattr(cudaq_target.cudaq, "set_target", _fake_set_target)

        result = cudaq_target.ensure_cudaq_target()

        assert set_target_calls == [("nvidia", "fp64")]
        assert result == "nvidia"

    def test_raises_without_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No compatible GPU must raise a clear error."""
        cudaq_target = _reset_and_import(monkeypatch)

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 0)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

        with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
            cudaq_target.ensure_cudaq_target()

    def test_raises_without_nvidia_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CUDA-Q install without NVIDIA target support must fail clearly."""
        cudaq_target = _reset_and_import(monkeypatch)

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: False)

        with pytest.raises(RuntimeError, match="NVIDIA target is unavailable"):
            cudaq_target.ensure_cudaq_target()


# ---------------------------------------------------------------------------
# Noisy target selection
# ---------------------------------------------------------------------------


class TestNoisyTarget:
    """Noisy (``noise_config.enabled=True``) target selection."""

    def test_gpu_trajectory_when_probe_succeeds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the GPU probe succeeds, noisy mode stays on ``nvidia``."""
        cudaq_target = _reset_and_import(monkeypatch)

        from solvers.noise import NoiseConfig

        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)

        # Simulate a successful probe by pre-setting the cache.
        monkeypatch.setattr(cudaq_target, "_gpu_noise_support", True)

        set_target_calls: list[tuple[str, str | None]] = []

        def _fake_set_target(name: str, option: str | None = None) -> None:
            set_target_calls.append((name, option))

        monkeypatch.setattr(cudaq_target.cudaq, "set_target", _fake_set_target)

        result = cudaq_target.ensure_cudaq_target(noise)

        assert result == "nvidia"
        assert set_target_calls == [("nvidia", "fp64")]

    def test_density_matrix_fallback_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the GPU probe fails, noisy mode falls back to ``density-matrix-cpu``."""
        cudaq_target = _reset_and_import(monkeypatch)

        from solvers.noise import NoiseConfig

        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)

        # Simulate a failed probe.
        monkeypatch.setattr(cudaq_target, "_gpu_noise_support", False)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

        set_target_calls: list[str] = []

        def _fake_set_target(name: str, **kwargs) -> None:
            set_target_calls.append(name)

        monkeypatch.setattr(cudaq_target.cudaq, "set_target", _fake_set_target)

        result = cudaq_target.ensure_cudaq_target(noise)

        assert result == "density-matrix-cpu"
        assert "density-matrix-cpu" in set_target_calls

    def test_raises_when_probe_fails_and_density_matrix_unavailable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No GPU trajectory AND no density-matrix-cpu must raise."""
        cudaq_target = _reset_and_import(monkeypatch)

        from solvers.noise import NoiseConfig

        noise = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)

        monkeypatch.setattr(cudaq_target, "_gpu_noise_support", False)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: False)

        with pytest.raises(RuntimeError, match="neither is available"):
            cudaq_target.ensure_cudaq_target(noise)


# ---------------------------------------------------------------------------
# Idempotency & helpers
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Idempotent target selection must not call ``set_target`` twice."""

    def test_second_call_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cudaq_target = _reset_and_import(monkeypatch)

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

        call_count = 0

        def _counting_set_target(name: str, option: str | None = None) -> None:
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr(cudaq_target.cudaq, "set_target", _counting_set_target)

        cudaq_target.ensure_cudaq_target()
        assert call_count == 1

        cudaq_target.ensure_cudaq_target()
        assert call_count == 1, "set_target should NOT be called a second time"

    def test_reset_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cudaq_target = _reset_and_import(monkeypatch)

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)

        call_count = 0

        def _counting_set_target(name: str, option: str | None = None) -> None:
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr(cudaq_target.cudaq, "set_target", _counting_set_target)

        cudaq_target.ensure_cudaq_target()
        assert call_count == 1

        cudaq_target.reset_target_state()

        cudaq_target.ensure_cudaq_target()
        assert call_count == 2, "set_target must be called again after reset"

    def test_get_current_target_reflects_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cudaq_target = _reset_and_import(monkeypatch)

        assert cudaq_target.get_current_target() is None

        monkeypatch.setattr(cudaq_target.cudaq, "num_available_gpus", lambda: 1)
        monkeypatch.setattr(cudaq_target.cudaq, "has_target", lambda name: True)
        monkeypatch.setattr(cudaq_target.cudaq, "set_target", lambda *a, **kw: None)

        cudaq_target.ensure_cudaq_target()
        assert cudaq_target.get_current_target() == "nvidia"

        cudaq_target.reset_target_state()
        assert cudaq_target.get_current_target() is None
