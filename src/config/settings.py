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
    """Project runtime settings loaded from environment variables.

    Attributes:
        quantum_backend: Selected solver backend name.
        output_dir: Root for experiment artifacts.
        input_dir: Project input data root.
        instance_config_path: Default YAML path for instance generation.
        enable_noise_simulation: When False, workflows may force noise off.
        random_seed: Default RNG seed for scripts that consume settings.

    """

    quantum_backend: BackendName
    output_dir: Path
    input_dir: Path
    instance_config_path: Path
    enable_noise_simulation: bool
    random_seed: int


def _project_root() -> Path:
    """Return the repository root (directory containing ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a ``.env``-style file.

    Args:
        path: File to read; missing files yield an empty dict.

    Returns:
        Mapping of variable names to unquoted string values.

    """
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
    """Resolve a setting: process environment overrides *values*, then *default*."""
    return os.getenv(key, values.get(key, default))


def _as_bool(value: str) -> bool:
    """Return True for common truthy string tokens (case-insensitive)."""
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    """Expand ``~`` and resolve relative paths against *project_root*."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (project_root / path)


def load_settings(env_file: Path | str | None = None, project_root: Path | None = None) -> Settings:
    """Load settings from `.env` and process environment variables.

    Args:
        env_file: Path to .env file. If None, uses project_root/.env.
        project_root: Project root for resolving relative paths. If None, inferred.

    Returns:
        Settings instance with quantum_backend, output_dir, input_dir, etc.

    Raises:
        ValueError: If HTSP_QUANTUM_BACKEND is unsupported.

    """
    resolved_root = project_root or _project_root()
    resolved_env_file = Path(env_file) if env_file else resolved_root / ".env"
    env_values = _read_env_file(resolved_env_file)

    backend = _get_setting("HTSP_QUANTUM_BACKEND", "simulated_annealing", env_values)
    if backend not in _VALID_BACKENDS:
        supported = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unsupported HTSP_QUANTUM_BACKEND '{backend}'. Expected one of: {supported}.")

    output_dir = _resolve_path(_get_setting("HTSP_OUTPUT_DIR", "output", env_values), resolved_root)
    input_dir = _resolve_path(_get_setting("HTSP_INPUT_DIR", "input", env_values), resolved_root)
    instance_config_path = _resolve_path(
        _get_setting(
            "HTSP_INSTANCE_CONFIG",
            "src/instance_gen_process/config.yaml",
            env_values,
        ),
        resolved_root,
    )
    enable_noise_simulation = _as_bool(
        _get_setting("HTSP_ENABLE_NOISE_SIMULATION", "false", env_values)
    )
    raw_seed = _get_setting("HTSP_RANDOM_SEED", "42", env_values)
    try:
        random_seed = int(raw_seed)
    except ValueError:
        raise ValueError(
            f"HTSP_RANDOM_SEED must be an integer, got {raw_seed!r}"
        ) from None

    valid_backends: dict[str, BackendName] = {
        "simulated_annealing": "simulated_annealing",
        "cirq": "cirq",
        "cudaq": "cudaq",
    }
    return Settings(
        quantum_backend=valid_backends[backend],
        output_dir=output_dir,
        input_dir=input_dir,
        instance_config_path=instance_config_path,
        enable_noise_simulation=enable_noise_simulation,
        random_seed=random_seed,
    )
