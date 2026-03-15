"""Tests for runtime settings loading."""

from pathlib import Path
import shutil
from uuid import uuid4

from config import load_settings


def _workspace_tmp_dir(prefix: str) -> Path:
    base_dir = Path(__file__).resolve().parent / ".tmp"
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def _cleanup_workspace_tmp_dir(temp_dir: Path) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)
    base_dir = temp_dir.parent
    if base_dir.exists() and not any(base_dir.iterdir()):
        base_dir.rmdir()


def test_load_settings_from_env_file() -> None:
    tmp_path = _workspace_tmp_dir("settings_env")
    env_path = tmp_path / ".env"
    try:
        env_path.write_text(
            "\n".join(
                [
                    "HTSP_QUANTUM_BACKEND=cirq",
                    "HTSP_OUTPUT_DIR=output",
                    "HTSP_INPUT_DIR=input_data",
                    "HTSP_INSTANCE_CONFIG=configs/instance.yaml",
                    "HTSP_ENABLE_NOISE_SIMULATION=true",
                    "HTSP_RANDOM_SEED=7",
                ]
            ),
            encoding="utf-8",
        )

        settings = load_settings(env_file=env_path, project_root=tmp_path)

        assert settings.quantum_backend == "cirq"
        assert settings.output_dir == tmp_path / "output"
        assert settings.input_dir == tmp_path / "input_data"
        assert settings.instance_config_path == tmp_path / "configs/instance.yaml"
        assert settings.enable_noise_simulation is True
        assert settings.random_seed == 7
    finally:
        _cleanup_workspace_tmp_dir(tmp_path)


def test_load_settings_defaults_without_env_file() -> None:
    tmp_path = _workspace_tmp_dir("settings_defaults")
    try:
        settings = load_settings(env_file=tmp_path / ".env.missing", project_root=tmp_path)

        assert settings.quantum_backend == "simulated_annealing"
        assert settings.output_dir == tmp_path / "output"
        assert settings.input_dir == tmp_path / "input"
        assert settings.instance_config_path == (tmp_path / "src/instance_gen_process/config.yaml")
        assert settings.enable_noise_simulation is False
        assert settings.random_seed == 42
    finally:
        _cleanup_workspace_tmp_dir(tmp_path)

