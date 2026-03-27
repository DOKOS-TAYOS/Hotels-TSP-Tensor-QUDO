"""Mean energy-history curves per QAOA depth (per-instance E/|E*| then mean over instances); excludes SA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np

from data_analysis._plot_typography import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_FONTSIZE_COMPACT,
    TICK_LABEL_FONTSIZE,
)


def _optimum_hline_y(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> float | None:
    """Y level of BF optimum after per-instance E/|E*| norm: ``sign(median ref)`` (±1)."""
    if paired.empty or "ref_objective_value" not in paired.columns:
        return None
    m = (
        paired["parse_ok"].astype(bool)
        & paired["solve_ok"].astype(bool)
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == int(n_cities))
    )
    sub = paired.loc[m, ["instance_key", "ref_objective_value"]].drop_duplicates(
        subset=["instance_key"],
    )
    vals = sub["ref_objective_value"].dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals != 0.0)]
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    if np.isfinite(med) and med != 0.0:
        return float(np.sign(med))
    pos = int(np.sum(vals > 0))
    neg = int(np.sum(vals < 0))
    if pos > neg:
        return 1.0
    if neg > pos:
        return -1.0
    return None


def _sorted_qaoa_depths(
    curves: Any,
    *,
    solver: str | None = None,
    formulation: str | None = None,
    n_cities: int | None = None,
) -> list[int]:
    """Distinct QAOA depths in ``curves`` for optional cohort filters."""
    import pandas as pd

    if curves.empty or "qaoa_depth" not in curves.columns:
        return []
    m = pd.Series(True, index=curves.index)
    if solver is not None:
        m &= curves["solver"] == solver
    if formulation is not None:
        m &= curves["formulation"] == formulation
    if n_cities is not None:
        m &= curves["n_cities"] == int(n_cities)
    qd = pd.to_numeric(curves.loc[m, "qaoa_depth"], errors="coerce")
    out: set[int] = set()
    for v in qd.dropna().tolist():
        f = float(v)
        if np.isfinite(f):
            out.add(int(f))
    return sorted(out)


def _mean_energy_curve_by_step(
    curves: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
    qaoa_depth: int | None = None,
) -> Any:
    """Mean objective trajectory vs optimizer step for one cohort.

    If ``qaoa_depth`` is set, restricts to that QAOA depth. Otherwise aggregates across
    depths at each step (weighted mean of per-depth means; same for ``std``).
    """
    import pandas as pd

    if curves.empty or "solver" not in curves.columns:
        return pd.DataFrame(columns=["step", "mean", "std"])
    m = (
        (curves["solver"] == solver)
        & (curves["formulation"] == formulation)
        & (curves["n_cities"] == n_cities)
    )
    if qaoa_depth is not None:
        qd = pd.to_numeric(curves["qaoa_depth"], errors="coerce")
        m &= qd == float(int(qaoa_depth))
    sub = curves.loc[m]
    if sub.empty:
        return pd.DataFrame(columns=["step", "mean", "std"])
    rows: list[dict[str, float | int]] = []
    for step, g in sub.groupby("step"):
        w = g["n_curves"].to_numpy(dtype=np.float64)
        y = g["mean"].to_numpy(dtype=np.float64)
        if "std" in g.columns:
            s = g["std"].fillna(0.0).to_numpy(dtype=np.float64)
        else:
            s = np.zeros(len(g), dtype=np.float64)
        den = float(np.sum(w))
        if den <= 0:
            continue
        rows.append(
            {
                "step": int(step),
                "mean": float(np.sum(y * w) / den),
                "std": float(np.sum(s * w) / den),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def _plot_mean_energy_with_std_band(
    ax: Any,
    df: Any,
    *,
    color: str,
    label: str,
    fill_alpha: float = 0.22,
) -> None:
    """Line of mean ± std band (values in ``df`` already per-instance normalized upstream)."""
    if df is None or getattr(df, "empty", True):
        return
    steps = df["step"].to_numpy(dtype=np.float64)
    mean = df["mean"].to_numpy(dtype=np.float64)
    std = df["std"].to_numpy(dtype=np.float64) if "std" in df.columns else np.zeros_like(mean)
    lo = mean - std
    hi = mean + std
    face = mcolors.to_rgba(color, alpha=fill_alpha)
    ax.fill_between(steps, lo, hi, facecolor=face, edgecolor="none", linewidth=0, zorder=1)
    ax.plot(steps, mean, color=color, linewidth=1.8, label=label, zorder=2)


def run_energy_history_figures(paired: Any, curves: Any, images_dir: Path) -> None:
    """Write mean energy curves per QAOA depth (``energy_curves_agg`` is per-instance E/|E*|)."""
    import matplotlib.pyplot as plt

    if curves is None or getattr(curves, "empty", True):
        return

    out = images_dir / "energy_history"
    out.mkdir(parents=True, exist_ok=True)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    y_label_norm = r"$f\,/\,|f^*|$ (mean ± $\sigma$)"

    depths_cudaq_n5 = sorted(
        set(_sorted_qaoa_depths(curves, solver="cudaq", formulation="qubo", n_cities=5))
        | set(
            _sorted_qaoa_depths(
                curves, solver="cudaq", formulation="tqudo_virtual", n_cities=5
            )
        )
    )
    n_cirq_cq_compare = (5, 9)
    depths_cirq_cudaq: set[int] = set()
    for n_cc in n_cirq_cq_compare:
        depths_cirq_cudaq |= set(
            _sorted_qaoa_depths(curves, solver="cirq", formulation="tqudo", n_cities=n_cc)
        )
        depths_cirq_cudaq |= set(
            _sorted_qaoa_depths(
                curves, solver="cudaq", formulation="tqudo_virtual", n_cities=n_cc
            )
        )

    # --- 1) CUDA-Q QUBO vs TQUDO qubits, n=5; one figure per p; single Y (normalized)
    for depth in depths_cudaq_n5:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        c_q = colors[0 % len(colors)]
        c_t = colors[1 % len(colors)]

        df_q = _mean_energy_curve_by_step(
            curves,
            solver="cudaq",
            formulation="qubo",
            n_cities=5,
            qaoa_depth=depth,
        )
        df_t = _mean_energy_curve_by_step(
            curves,
            solver="cudaq",
            formulation="tqudo_virtual",
            n_cities=5,
            qaoa_depth=depth,
        )
        plot_q = not df_q.empty
        plot_t = not df_t.empty
        if not plot_q and not plot_t:
            plt.close(fig)
            continue

        if plot_q:
            _plot_mean_energy_with_std_band(ax, df_q, color=c_q, label="QUBO")
            yq = _optimum_hline_y(paired, solver="cudaq", formulation="qubo", n_cities=5)
            if yq is not None:
                ax.axhline(yq, color=c_q, linestyle="--", linewidth=1.2, alpha=0.88)
        if plot_t:
            _plot_mean_energy_with_std_band(ax, df_t, color=c_t, label="TQUDO qubits")
            yt = _optimum_hline_y(
                paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5
            )
            if yt is not None:
                ax.axhline(yt, color=c_t, linestyle="--", linewidth=1.2, alpha=0.88)

        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(y_label_norm, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")
        fig.tight_layout()
        fig.savefig(
            out / f"cudaq_qubo_tvirt_n5_p{depth}.png",
            dpi=150,
        )
        plt.close(fig)

    # --- 2) TQUDO qudits (Cirq) vs TQUDO qubits (CUDA-Q): n=5 and n=9 as four series ---
    for stale in out.glob("cirq_tqudo_vs_cq_tvirt_n5_p*.png"):
        if stale.is_file():
            stale.unlink()
    series_cq_cirq: tuple[tuple[str, str, str], ...] = (
        ("cirq", "tqudo", "TQUDO qudits"),
        ("cudaq", "tqudo_virtual", "TQUDO qubits"),
    )
    for depth in sorted(depths_cirq_cudaq):
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        color_idx = 0
        for n_cc in n_cirq_cq_compare:
            for solver, formulation, base_label in series_cq_cirq:
                df = _mean_energy_curve_by_step(
                    curves,
                    solver=solver,
                    formulation=formulation,
                    n_cities=n_cc,
                    qaoa_depth=depth,
                )
                if df.empty:
                    continue
                c = colors[color_idx % len(colors)]
                color_idx += 1
                lab = f"{base_label}, n = {n_cc}"
                _plot_mean_energy_with_std_band(ax, df, color=c, label=lab)
                yn = _optimum_hline_y(
                    paired, solver=solver, formulation=formulation, n_cities=n_cc
                )
                if yn is not None:
                    ax.axhline(yn, color=c, linestyle="--", linewidth=1.2, alpha=0.88)
                plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(y_label_norm, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE_COMPACT, loc="best")
        fig.tight_layout()
        fig.savefig(
            out / f"cirq_tqudo_vs_cq_tvirt_n5_n9_p{depth}.png",
            dpi=150,
        )
        plt.close(fig)

    # --- 3) TQUDO qudits (Cirq): all n_cities on one axes per QAOA depth ---
    cirq_t = curves[(curves["solver"] == "cirq") & (curves["formulation"] == "tqudo")]
    if cirq_t.empty:
        return
    n_list = sorted({int(x) for x in cirq_t["n_cities"].dropna().tolist()})
    if not n_list:
        return
    depths_by_n = sorted(
        {int(d) for d in _sorted_qaoa_depths(curves, solver="cirq", formulation="tqudo")}
    )
    for depth in depths_by_n:
        fig, ax = plt.subplots(figsize=(9, 5))
        any_line = False
        for i, n_int in enumerate(n_list):
            df = _mean_energy_curve_by_step(
                curves,
                solver="cirq",
                formulation="tqudo",
                n_cities=n_int,
                qaoa_depth=depth,
            )
            if df.empty:
                continue
            c = colors[i % len(colors)]
            _plot_mean_energy_with_std_band(
                ax, df, color=c, label=f"n = {n_int}"
            )
            yn = _optimum_hline_y(
                paired, solver="cirq", formulation="tqudo", n_cities=n_int
            )
            if yn is not None:
                ax.axhline(yn, color=c, linestyle="--", linewidth=1.2, alpha=0.88)
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(y_label_norm, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE_COMPACT, loc="best")
        fig.tight_layout()
        fig.savefig(
            out / f"cirq_tqudo_by_n_p{depth}.png",
            dpi=150,
        )
        plt.close(fig)
