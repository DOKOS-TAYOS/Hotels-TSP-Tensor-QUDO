"""Streamlit shell for reproducible experiment configuration."""

from __future__ import annotations

import streamlit as st

from config import load_settings
from instance_gen_process import load_instance_config


def main() -> None:
    """Render the initial project dashboard."""

    st.set_page_config(page_title="Aircraft Loading Tensor-QUDO", layout="wide")
    st.title("Aircraft Loading Problem Tensor-QUDO")
    st.caption("Scaffold UI for reproducible experiment setup and future solver runs.")

    settings = load_settings()
    instance_config = load_instance_config(settings.instance_config_path)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Runtime settings")
        st.json(
            {
                "quantum_backend": settings.quantum_backend,
                "output_dir": str(settings.output_dir),
                "input_dir": str(settings.input_dir),
                "instance_config_path": str(settings.instance_config_path),
                "enable_noise_simulation": settings.enable_noise_simulation,
                "random_seed": settings.random_seed,
            }
        )

    with right_col:
        st.subheader("Instance generation config")
        st.json(
            {
                "num_items": instance_config.num_items,
                "max_weight": instance_config.max_weight,
                "max_volume": instance_config.max_volume,
                "cg_min": instance_config.cg_min,
                "cg_max": instance_config.cg_max,
                "weight_range": instance_config.weight_range,
                "volume_range": instance_config.volume_range,
                "seed": instance_config.seed,
            }
        )

    st.info("Solver backends are scaffolded and ready for algorithm implementation.")


if __name__ == "__main__":
    main()

