"""Streamlit shell for reproducible experiment configuration."""

from __future__ import annotations

from types import ModuleType

from config import load_settings
from instance_gen_process import load_instance_config


def _load_streamlit() -> ModuleType:
    """Import Streamlit lazily so importing the package works without the UI extra.

    Returns:
        The ``streamlit`` module.

    Raises:
        ModuleNotFoundError: If Streamlit is not installed (hint to use ``[ui]``).
    """
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "streamlit_app requires the optional 'ui' dependencies. "
            "Install them with `pip install -e .[ui]` or run `./install.sh dev,ui,cudaq`."
        ) from exc
    return st


def main() -> None:
    """Configure the Streamlit page and show runtime plus instance YAML summaries."""
    st = _load_streamlit()

    st.set_page_config(page_title="Hotel TSP · Tensor-QUDO", layout="wide")
    st.title("Hotel TSP · Tensor-QUDO y QUBO")
    st.caption(
        "Routing con restricciones de precedencia: formulación QUBO (one-hot) frente a Tensor-QUDO "
        "(qudits). Referencia: arXiv:2508.01958. Configuración local antes de lanzar pipelines "
        "(`experiments`, `data_analysis`)."
    )

    settings = load_settings()
    instance_config = load_instance_config(settings.instance_config_path)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Entorno y salida")
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
        st.subheader("Generación de instancias")
        st.json(
            {
                "n_cities": instance_config.n_cities,
                "n_precedences_range": list(instance_config.n_precedences_range),
                "prices_range_hotels": list(instance_config.prices_range_hotels),
                "prices_range_travels": list(instance_config.prices_range_travels),
                "seed": instance_config.seed,
            }
        )

    st.info(
        "Los resultados agregados y figuras se exploran en `webpage_results/` "
        "(servidor HTTP: `make -f scripts/makefile results-web`). "
        "Backends: CUDA-Q, Cirq, recocido simulado y fuerza bruta según `solver_config.yaml`."
    )


if __name__ == "__main__":
    main()
