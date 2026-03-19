"""Smoke tests for package imports."""

from config import Settings, load_settings
from instance_gen_process import (
    InstanceConfig,
    ProblemInstance,
    generate_random_instance,
    load_instance_config,
)
from solvers import (
    CirqSolver,
    CudaqSolver,
    SimulatedAnnealingSolver,
    SolverResult,
    SolverRunConfig,
)
from streamlit_app import main as streamlit_main
from utils import build_output_layout, configure_logging, validate_instance_constraints


def test_import_smoke() -> None:
    assert Settings is not None
    assert load_settings is not None
    assert load_instance_config is not None
    assert InstanceConfig is not None
    assert ProblemInstance is not None
    assert generate_random_instance is not None
    assert SolverRunConfig is not None
    assert SolverResult is not None
    assert CirqSolver is not None
    assert CudaqSolver is not None
    assert SimulatedAnnealingSolver is not None
    assert build_output_layout is not None
    assert configure_logging is not None
    assert validate_instance_constraints is not None
    assert streamlit_main is not None
