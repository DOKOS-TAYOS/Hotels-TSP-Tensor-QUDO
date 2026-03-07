"""Tests for instance config loading and generation."""

from pathlib import Path
import shutil
from uuid import uuid4

from instance_gen_process import generate_random_instance, load_instance_config


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


def test_load_instance_config_and_generate() -> None:
    tmp_path = _workspace_tmp_dir("instance_config")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "num_items: 3",
                    "max_weight: 1000",
                    "max_volume: 80",
                    "cg_min: -5",
                    "cg_max: 5",
                    "weight_range: [10, 20]",
                    "volume_range: [1, 3]",
                    "seed: 123",
                ]
            ),
            encoding="utf-8",
        )

        config = load_instance_config(config_path)
        instance = generate_random_instance(config)

        assert config.num_items == 3
        assert len(instance.items) == 3
        assert instance.max_weight == 1000
        assert instance.max_volume == 80
        assert instance.cg_min == -5
        assert instance.cg_max == 5
    finally:
        _cleanup_workspace_tmp_dir(tmp_path)

