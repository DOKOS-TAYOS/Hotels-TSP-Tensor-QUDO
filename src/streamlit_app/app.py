"""Streamlit shell for reproducible experiment configuration."""

from __future__ import annotations

from config import load_settings
from instance_gen_process import load_instance_config


def _load_streamlit():
    """Import Streamlit lazily so the package remains importable without the UI extra."""
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "streamlit_app requires the optional 'ui' dependencies. "
            "Install them with `pip install -e .[ui]` or run `./install.sh dev,ui,cudaq`."
        ) from exc
    return st


def main() -> None:
    """Render the initial project dashboard."""
    st = _load_streamlit()

    st.set_page_config(page_title="Hotel TSP Tensor-QUDO", layout="wide")
    st.title("Hotel TSP with Tensor-QUDO")
    st.caption(
        "Travel routing optimization with precedence constraints. "
        "Scaffold UI for reproducible experiment setup and future solver runs."
    )

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
                "n_cities": instance_config.n_cities,
                "n_precedences_range": list(instance_config.n_precedences_range),
                "prices_range_hotels": list(instance_config.prices_range_hotels),
                "prices_range_travels": list(instance_config.prices_range_travels),
                "seed": instance_config.seed,
            }
        )

    st.info("Solver backends are scaffolded and ready for algorithm implementation.")


if __name__ == "__main__":
    main()
