"""Tests for CUDA-Q parallel experiment scheduling helpers."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from experiments.cudaq_parallel import (
    CIRQ_PARALLEL_ENV,
    CUDAQ_PARALLEL_ENV,
    CudaqParallelJobSpec,
    _compact_parallel_status_line,
    resolve_cirq_max_parallel_instances,
    resolve_cudaq_max_parallel_instances,
    run_cudaq_parallel_batch,
)
from experiments.workflow_io import serialize_problem_instance
from instance_gen_process.solver_config_loader import (
    parse_solver_config_dict,
    solver_config_to_run_config,
)
from conftest import make_problem_instance


def test_compact_parallel_status_line_parses_inst_suffix() -> None:
    """Regression: pattern must match ``... inst=k`` (not ``ins=``)."""
    labels = frozenset(
        {
            "n_cities=5 depth=1 inst=2",
            "n_cities=5 depth=1 inst=7",
        }
    )
    line = _compact_parallel_status_line("[parallel cudaq]", labels, 3, 100, 120)
    assert "active_inst=[2,7]" in line
    assert "writes=3/100" in line


def test_resolve_cudaq_max_parallel_instances_default() -> None:
    assert resolve_cudaq_max_parallel_instances({}) == 1
    assert resolve_cudaq_max_parallel_instances({"cudaq_max_parallel_instances": 4}) == 4


def test_resolve_cudaq_max_parallel_instances_clamps_minimum() -> None:
    assert resolve_cudaq_max_parallel_instances({"cudaq_max_parallel_instances": 0}) == 1
    assert resolve_cudaq_max_parallel_instances({"cudaq_max_parallel_instances": -9}) == 1


def test_resolve_cudaq_max_parallel_instances_env_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CUDAQ_PARALLEL_ENV, "8")
    assert resolve_cudaq_max_parallel_instances({"cudaq_max_parallel_instances": 2}) == 8
    monkeypatch.delenv(CUDAQ_PARALLEL_ENV, raising=False)


def test_resolve_cudaq_max_parallel_instances_env_empty_uses_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CUDAQ_PARALLEL_ENV, "   ")
    assert resolve_cudaq_max_parallel_instances({"cudaq_max_parallel_instances": 3}) == 3
    monkeypatch.delenv(CUDAQ_PARALLEL_ENV, raising=False)


def test_resolve_cirq_max_parallel_instances_default() -> None:
    assert resolve_cirq_max_parallel_instances({}) == 1
    assert resolve_cirq_max_parallel_instances({"cirq_max_parallel_instances": 4}) == 4


def test_resolve_cirq_max_parallel_instances_clamps_minimum() -> None:
    assert resolve_cirq_max_parallel_instances({"cirq_max_parallel_instances": 0}) == 1
    assert resolve_cirq_max_parallel_instances({"cirq_max_parallel_instances": -2}) == 1


def test_resolve_cirq_max_parallel_instances_env_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CIRQ_PARALLEL_ENV, "6")
    assert resolve_cirq_max_parallel_instances({"cirq_max_parallel_instances": 2}) == 6
    monkeypatch.delenv(CIRQ_PARALLEL_ENV, raising=False)


def test_resolve_cirq_max_parallel_instances_env_empty_uses_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CIRQ_PARALLEL_ENV, "  ")
    assert resolve_cirq_max_parallel_instances({"cirq_max_parallel_instances": 3}) == 3
    monkeypatch.delenv(CIRQ_PARALLEL_ENV, raising=False)


def test_run_cudaq_parallel_batch_empty() -> None:
    def _never_called(_job: object, _payload: object) -> Path:
        raise AssertionError("should not write")

    r = run_cudaq_parallel_batch(
        [],
        max_workers=4,
        solutions_write_fn=_never_called,
        is_interrupted=lambda: False,
    )
    assert r.n_failed == 0
    assert r.n_completed == 0
    assert not r.interrupted


def test_run_parallel_batch_rejects_mixed_solvers(tmp_path: Path) -> None:
    """All job specs in one batch must share solver_name."""
    inst = make_problem_instance(n_cities=5)
    inst_path = tmp_path / "i.json"
    inst_path.write_text(json.dumps(serialize_problem_instance(inst)), encoding="utf-8")
    base_cfg = {
        "n_instances": 2,
        "solver": "cirq",
        "formulation": "qubo",
        "optimizer": "COBYLA",
        "restriction": {"lambda_0": 100.0, "lambda_1": 100.0, "lambda_2": 100.0},
        "qaoa_depth": 1,
        "qaoa_max_iter": 8,
        "qaoa_delta_t": 0.55,
        "qaoa_optimizer_tol": 1.0e-6,
        "qaoa_shots": 32,
        "qaoa_sample_shots": 32,
        "seed": 0,
        "max_iterations": 100,
        "timeout_seconds": None,
        "sa_t_initial": 1000.0,
        "sa_t_final": 1.0e-6,
        "sa_alpha": 0.995,
        "noise": {"enabled": False},
    }
    validated = parse_solver_config_dict(base_cfg)
    run_config = solver_config_to_run_config(validated)
    common = {
        "instance_json_path": str(inst_path),
        "status_label": "x",
        "run_config": run_config,
        "instance_config_dict": {},
        "solver_config_serializable": {},
        "formulation": "qubo",
        "n_cities": 5,
        "path_depth": 1,
        "output_root": str(tmp_path),
    }
    specs = [
        CudaqParallelJobSpec(k=1, solver_name="cirq", **common),
        CudaqParallelJobSpec(k=2, solver_name="cudaq", **common),
    ]
    with pytest.raises(ValueError, match="same solver_name"):
        run_cudaq_parallel_batch(
            specs,
            max_workers=2,
            solutions_write_fn=lambda _j, _p: tmp_path / "x.json",
            is_interrupted=lambda: False,
        )


class _InlineProcessPoolExecutor:
    """Runs submitted callables in-process (for testing without subprocess spawn)."""

    def __init__(self, max_workers: int = 1, mp_context: Any = None) -> None:
        pass

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            fut.set_result(fn(*args))
        except Exception as exc:
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


def test_run_cirq_parallel_batch_two_instances_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cirq path with inline executor so CirqSolver can be mocked in-process."""
    import experiments.cudaq_parallel as cqp

    monkeypatch.setattr(cqp, "ProcessPoolExecutor", _InlineProcessPoolExecutor)

    inst_dir = tmp_path / "raw" / "instances" / "n_5"
    inst_dir.mkdir(parents=True)
    for k in (1, 2):
        inst = make_problem_instance(n_cities=5)
        (inst_dir / f"instance_{k}.json").write_text(
            json.dumps(serialize_problem_instance(inst)),
            encoding="utf-8",
        )

    base_cfg = {
        "n_instances": 2,
        "solver": "cirq",
        "formulation": "qubo",
        "optimizer": "COBYLA",
        "restriction": {"lambda_0": 100.0, "lambda_1": 100.0, "lambda_2": 100.0},
        "qaoa_depth": 1,
        "qaoa_max_iter": 8,
        "qaoa_delta_t": 0.55,
        "qaoa_optimizer_tol": 1.0e-6,
        "qaoa_shots": 32,
        "qaoa_sample_shots": 32,
        "seed": 0,
        "max_iterations": 100,
        "timeout_seconds": None,
        "sa_t_initial": 1000.0,
        "sa_t_final": 1.0e-6,
        "sa_alpha": 0.995,
        "noise": {"enabled": False},
    }
    validated = parse_solver_config_dict(base_cfg)
    run_config = solver_config_to_run_config(validated)
    serializable = {
        "n_instances": validated["n_instances"],
        "solver": validated["solver"],
        "formulation": validated["formulation"],
        "cirq_max_parallel_instances_effective": 2,
    }

    out_root = str(tmp_path.resolve())
    specs = [
        CudaqParallelJobSpec(
            k=k,
            instance_json_path=str(inst_dir / f"instance_{k}.json"),
            status_label=f"n_cities=5 depth=1 inst={k}",
            run_config=run_config,
            instance_config_dict={"n_cities": 5, "seed": 0},
            solver_config_serializable=serializable,
            solver_name="cirq",
            formulation="qubo",
            n_cities=5,
            path_depth=1,
            output_root=out_root,
        )
        for k in (1, 2)
    ]

    written: list[Path] = []

    def _write(job: object, payload: dict[str, Any]) -> Path:
        from experiments.cudaq_parallel import CudaqParallelJob

        assert isinstance(job, CudaqParallelJob)
        out = tmp_path / "out" / f"instance_{job.k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload), encoding="utf-8")
        written.append(out)
        return out

    from solvers.base import SolverResult

    mock_solver = MagicMock()
    mock_solver.solve.return_value = SolverResult(
        solver_name="cirq",
        objective_value=0.0,
        feasible=True,
        runtime_seconds=0.01,
        metadata={},
    )

    with patch("solvers.cirq_solver.CirqSolver", return_value=mock_solver):
        batch = run_cudaq_parallel_batch(
            specs,
            max_workers=2,
            solutions_write_fn=_write,
            is_interrupted=lambda: False,
        )

    assert batch.n_completed == 2
    assert batch.n_failed == 0
    assert not batch.interrupted
    assert len(written) == 2
    assert mock_solver.solve.call_count == 2


@pytest.mark.skipif(
    os.environ.get("HTSP_RUN_CUDA_PARALLEL_GPU_TEST") != "1",
    reason="Set HTSP_RUN_CUDA_PARALLEL_GPU_TEST=1 to run real GPU parallel batch smoke test",
)
def test_cudaq_parallel_batch_two_instances_smoke_gpu(tmp_path: Path) -> None:
    """Two QUBO n=5 instances, 2 workers; requires CUDA-Q + NVIDIA target."""
    cudaq = pytest.importorskip("cudaq")
    if cudaq.num_available_gpus() < 1 or not cudaq.has_target("nvidia"):
        pytest.skip("No CUDA-Q NVIDIA GPU")

    inst_dir = tmp_path / "raw" / "instances" / "n_5"
    inst_dir.mkdir(parents=True)
    for k in (1, 2):
        inst = make_problem_instance(n_cities=5)
        (inst_dir / f"instance_{k}.json").write_text(
            json.dumps(serialize_problem_instance(inst)),
            encoding="utf-8",
        )

    base_cfg = {
        "n_instances": 2,
        "solver": "cudaq",
        "formulation": "qubo",
        "optimizer": "COBYLA",
        "restriction": {"lambda_0": 100.0, "lambda_1": 100.0, "lambda_2": 100.0},
        "qaoa_depth": 1,
        "qaoa_max_iter": 8,
        "qaoa_delta_t": 0.55,
        "qaoa_optimizer_tol": 1.0e-6,
        "qaoa_shots": 32,
        "qaoa_sample_shots": 32,
        "seed": 0,
        "max_iterations": 100,
        "timeout_seconds": None,
        "sa_t_initial": 1000.0,
        "sa_t_final": 1.0e-6,
        "sa_alpha": 0.995,
        "noise": {"enabled": False},
    }
    validated = parse_solver_config_dict(base_cfg)
    run_config = solver_config_to_run_config(validated)
    serializable = {
        "n_instances": validated["n_instances"],
        "solver": validated["solver"],
        "formulation": validated["formulation"],
        "cudaq_max_parallel_instances_effective": 2,
    }

    out_root = str(tmp_path.resolve())
    specs = [
        CudaqParallelJobSpec(
            k=k,
            instance_json_path=str(inst_dir / f"instance_{k}.json"),
            status_label=f"n_cities=5 depth=1 inst={k}",
            run_config=run_config,
            instance_config_dict={"n_cities": 5, "seed": 0},
            solver_config_serializable=serializable,
            solver_name="cudaq",
            formulation="qubo",
            n_cities=5,
            path_depth=1,
            output_root=out_root,
        )
        for k in (1, 2)
    ]

    written: list[Path] = []

    def _write(job: object, payload: dict) -> Path:
        from experiments.cudaq_parallel import CudaqParallelJob

        assert isinstance(job, CudaqParallelJob)
        out = tmp_path / "out" / f"instance_{job.k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload), encoding="utf-8")
        written.append(out)
        return out

    batch = run_cudaq_parallel_batch(
        specs,
        max_workers=2,
        solutions_write_fn=_write,
        is_interrupted=lambda: False,
    )
    assert batch.n_completed == 2
    assert batch.n_failed == 0
    assert not batch.interrupted
    assert len(written) == 2
    for p in written:
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "solver_output" in data
        assert "error" not in data.get("solver_output", {})


def test_run_experiment_parallel_path_not_used_when_workers_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cudaq_max_parallel_instances=1, ProcessPoolExecutor must not start."""
    import experiments.cudaq_parallel as cqp
    from experiments import main_experiment_workflow as mew

    called: list[int] = []

    class BoomExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            called.append(1)
            raise AssertionError("ProcessPoolExecutor should not be constructed")

        def submit(self, *_a: object, **_k: object) -> object:
            raise AssertionError

        def shutdown(self, *_a: object, **_k: object) -> None:
            pass

    monkeypatch.setattr(cqp, "ProcessPoolExecutor", BoomExecutor)

    inst = make_problem_instance(n_cities=5)
    inst_dir = tmp_path / "raw" / "instances" / "n_5"
    inst_dir.mkdir(parents=True)
    for k in (1, 2):
        (inst_dir / f"instance_{k}.json").write_text(
            json.dumps(serialize_problem_instance(inst)),
            encoding="utf-8",
        )

    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        "\n".join(
            [
                "n_cities: 5",
                "n_instances: 2",
                "solver: cudaq",
                "formulation: qubo",
                "qaoa_depth: 1",
                "qaoa_max_iter: 8",
                "qaoa_shots: 4",
                "qaoa_sample_shots: 4",
                "cudaq_max_parallel_instances: 1",
                "optimizer: COBYLA",
                "restriction:",
                "  lambda_0: 100.0",
                "  lambda_1: 100.0",
                "  lambda_2: 100.0",
            ]
        ),
        encoding="utf-8",
    )

    base_solver = tmp_path / "base.yaml"
    base_solver.write_text(
        "\n".join(
            [
                "n_instances: 2",
                "solver: cudaq",
                "formulation: qubo",
                "optimizer: COBYLA",
                "restriction:",
                "  lambda_0: 100.0",
                "  lambda_1: 100.0",
                "  lambda_2: 100.0",
                "qaoa_depth: 1",
                "qaoa_max_iter: 8",
                "qaoa_delta_t: 0.55",
                "qaoa_optimizer_tol: 1.0e-6",
                "qaoa_shots: 4",
                "qaoa_sample_shots: 4",
                "seed: 0",
                "max_iterations: 100",
                "timeout_seconds: null",
                "sa_t_initial: 1000.0",
                "sa_t_final: 1.0e-6",
                "sa_alpha: 0.995",
                "noise:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    icfg = tmp_path / "icfg.yaml"
    icfg.write_text(
        "\n".join(
            [
                "n_cities: 5",
                "n_precedences_range: [0, 0]",
                "prices_range_hotels: [1.0, 1.0]",
                "prices_range_travels: [1.0, 1.0]",
                "seed: 0",
            ]
        ),
        encoding="utf-8",
    )

    with patch.object(mew.CudaqSolver, "solve") as mock_solve:
        from solvers.base import SolverResult

        mock_solve.return_value = SolverResult(
            solver_name="cudaq",
            objective_value=0.0,
            feasible=True,
            runtime_seconds=0.01,
            metadata={},
        )
        mew.run_experiment_from_yaml(
            experiment_yaml_path=exp_yaml,
            instance_config_path=icfg,
            solver_config_path=base_solver,
            output_root=tmp_path,
            settings=None,
        )

    assert called == []
    assert mock_solve.call_count == 2


def test_run_experiment_cirq_parallel_path_not_used_when_workers_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cirq_max_parallel_instances=1, ProcessPoolExecutor must not start."""
    import experiments.cudaq_parallel as cqp
    from experiments import main_experiment_workflow as mew

    called: list[int] = []

    class BoomExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            called.append(1)
            raise AssertionError("ProcessPoolExecutor should not be constructed")

        def submit(self, *_a: object, **_k: object) -> object:
            raise AssertionError

        def shutdown(self, *_a: object, **_k: object) -> None:
            pass

    monkeypatch.setattr(cqp, "ProcessPoolExecutor", BoomExecutor)

    inst = make_problem_instance(n_cities=5)
    inst_dir = tmp_path / "raw" / "instances" / "n_5"
    inst_dir.mkdir(parents=True)
    for k in (1, 2):
        (inst_dir / f"instance_{k}.json").write_text(
            json.dumps(serialize_problem_instance(inst)),
            encoding="utf-8",
        )

    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        "\n".join(
            [
                "n_cities: 5",
                "n_instances: 2",
                "solver: cirq",
                "formulation: qubo",
                "qaoa_depth: 1",
                "qaoa_max_iter: 8",
                "qaoa_shots: 4",
                "qaoa_sample_shots: 4",
                "cirq_max_parallel_instances: 1",
                "optimizer: COBYLA",
                "restriction:",
                "  lambda_0: 100.0",
                "  lambda_1: 100.0",
                "  lambda_2: 100.0",
            ]
        ),
        encoding="utf-8",
    )

    base_solver = tmp_path / "base.yaml"
    base_solver.write_text(
        "\n".join(
            [
                "n_instances: 2",
                "solver: cirq",
                "formulation: qubo",
                "optimizer: COBYLA",
                "restriction:",
                "  lambda_0: 100.0",
                "  lambda_1: 100.0",
                "  lambda_2: 100.0",
                "qaoa_depth: 1",
                "qaoa_max_iter: 8",
                "qaoa_delta_t: 0.55",
                "qaoa_optimizer_tol: 1.0e-6",
                "qaoa_shots: 4",
                "qaoa_sample_shots: 4",
                "seed: 0",
                "max_iterations: 100",
                "timeout_seconds: null",
                "sa_t_initial: 1000.0",
                "sa_t_final: 1.0e-6",
                "sa_alpha: 0.995",
                "noise:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    icfg = tmp_path / "icfg.yaml"
    icfg.write_text(
        "\n".join(
            [
                "n_cities: 5",
                "n_precedences_range: [0, 0]",
                "prices_range_hotels: [1.0, 1.0]",
                "prices_range_travels: [1.0, 1.0]",
                "seed: 0",
            ]
        ),
        encoding="utf-8",
    )

    with patch.object(mew.CirqSolver, "solve") as mock_solve:
        from solvers.base import SolverResult

        mock_solve.return_value = SolverResult(
            solver_name="cirq",
            objective_value=0.0,
            feasible=True,
            runtime_seconds=0.01,
            metadata={},
        )
        mew.run_experiment_from_yaml(
            experiment_yaml_path=exp_yaml,
            instance_config_path=icfg,
            solver_config_path=base_solver,
            output_root=tmp_path,
            settings=None,
        )

    assert called == []
    assert mock_solve.call_count == 2
