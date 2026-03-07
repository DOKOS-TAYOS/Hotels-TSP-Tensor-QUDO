"""Runtime settings for local and reproducible experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal

BackendName = Literal["simulated_annealing", "cirq", "cudaq"]
_VALID_BACKENDS: set[str] = {"simulated_annealing", "cirq", "cudaq"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Project runtime settings loaded from environment variables."""

    quantum_backend: BackendName
    output_dir: Path
    input_dir: Path
    instance_config_path: Path
    enable_noise_simulation: bool
    random_seed: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        values[key.strip()] = raw_value.strip().strip("\"'")
    return values


def _get_setting(key: str, default: str, values: dict[str, str]) -> str:
    return os.getenv(key, values.get(key, default))


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (project_root / path)


def load_settings(env_file: Path | str | None = None, project_root: Path | None = None) -> Settings:
    """Load settings from `.env` and process environment variables."""

    resolved_root = project_root or _project_root()
    resolved_env_file = Path(env_file) if env_file else resolved_root / ".env"
    env_values = _read_env_file(resolved_env_file)

    backend = _get_setting("ALP_QUANTUM_BACKEND", "simulated_annealing", env_values)
    if backend not in _VALID_BACKENDS:
        supported = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unsupported ALP_QUANTUM_BACKEND '{backend}'. Expected one of: {supported}.")

    output_dir = _resolve_path(_get_setting("ALP_OUTPUT_DIR", "output", env_values), resolved_root)
    input_dir = _resolve_path(_get_setting("ALP_INPUT_DIR", "input", env_values), resolved_root)
    instance_config_path = _resolve_path(
        _get_setting(
            "ALP_INSTANCE_CONFIG",
            "src/instance_gen_process/config.yaml",
            env_values,
        ),
        resolved_root,
    )
    enable_noise_simulation = _as_bool(
        _get_setting("ALP_ENABLE_NOISE_SIMULATION", "false", env_values)
    )
    random_seed = int(_get_setting("ALP_RANDOM_SEED", "42", env_values))

    return Settings(
        quantum_backend=backend,  # type: ignore[arg-type]
        output_dir=output_dir,
        input_dir=input_dir,
        instance_config_path=instance_config_path,
        enable_noise_simulation=enable_noise_simulation,
        random_seed=random_seed,
    )
