"""Mean energy-history curves with brute-force optimal objective (analysis excludes SA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np


def _mean_energy_curve_by_step(
    curves: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> Any:
    """Weighted mean and combined spread over QAOA depths at each optimizer step.

    Per-depth ``std`` is sample std (ddof=1) of curves at that step. Across depths,
    combined band uses ``sum(w * std) / sum(w)`` (same weights ``n_curves`` as the mean).
    """
    import pandas as pd

    if curves.empty or "solver" not in curves.columns:
        return pd.DataFrame(columns=["step", "mean", "std"])
    m = (
        (curves["solver"] == solver)
        & (curves["formulation"] == formulation)
        & (curves["n_cities"] == n_cities)
    )
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
    """Line of mean energy and semitransparent ``mean ± std`` band."""
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


def _mean_ref_objective(
    paired: Any,
    *,
    solver: str,
    formulation: str,
    n_cities: int,
) -> float | None:
    """Mean ``ref_objective_value`` over distinct ``instance_key`` (requires brute-force ref.)."""
    if paired.empty or "ref_objective_value" not in paired.columns:
        return None
    m = (
        paired["parse_ok"].astype(bool)
        & paired["solve_ok"].astype(bool)
        & (paired["solver"] == solver)
        & (paired["formulation"] == formulation)
        & (paired["n_cities"] == n_cities)
    )
    sub = paired.loc[m, ["instance_key", "ref_objective_value"]].drop_duplicates(
        subset=["instance_key"],
    )
    vals = sub["ref_objective_value"].dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _mean_ref_tqudo_shared_n(
    paired: Any,
    *,
    n_cities: int,
) -> float | None:
    """Single BF TQUDO objective reference: cohort = Cirq native + CUDA-Q virtual on same n."""
    if paired.empty or "ref_objective_value" not in paired.columns:
        return None
    m = (
        paired["parse_ok"].astype(bool)
        & paired["solve_ok"].astype(bool)
        & (paired["n_cities"] == n_cities)
        & (
            ((paired["solver"] == "cirq") & (paired["formulation"] == "tqudo"))
            | ((paired["solver"] == "cudaq") & (paired["formulation"] == "tqudo_virtual"))
        )
    )
    sub = paired.loc[m, ["instance_key", "ref_objective_value"]].drop_duplicates(
        subset=["instance_key"],
    )
    vals = sub["ref_objective_value"].dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def run_energy_history_figures(paired: Any, curves: Any, images_dir: Path) -> None:
    """Write mean energy curves (excluding SA from processed tables upstream)."""
    import matplotlib.pyplot as plt

    if curves is None or getattr(curves, "empty", True):
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]

    # --- 1) CUDA-Q QUBO vs CUDA-Q TQUDO virtual, n=5; separate Y scales (QUBO left, TQUDO right) ---
    fig, ax_q = plt.subplots(figsize=(8, 4.5))
    ax_t = ax_q.twinx()
    c_q = colors[0 % len(colors)]
    c_t = colors[1 % len(colors)]

    df_q = _mean_energy_curve_by_step(
        curves, solver="cudaq", formulation="qubo", n_cities=5
    )
    _plot_mean_energy_with_std_band(
        ax_q, df_q, color=c_q, label="QUBO"
    )
    ref_q = _mean_ref_objective(
        paired, solver="cudaq", formulation="qubo", n_cities=5
    )
    if ref_q is not None:
        ax_q.axhline(ref_q, color=c_q, linestyle="--", linewidth=1.2, alpha=0.88)

    df_t = _mean_energy_curve_by_step(
        curves, solver="cudaq", formulation="tqudo_virtual", n_cities=5
    )
    _plot_mean_energy_with_std_band(
        ax_t, df_t, color=c_t, label="TQUDO virt."
    )
    ref_t = _mean_ref_objective(
        paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5
    )
    if ref_t is not None:
        ax_t.axhline(ref_t, color=c_t, linestyle="--", linewidth=1.2, alpha=0.88)

    ax_q.set_xlabel("Step")
    ax_q.set_ylabel("QUBO", color=c_q)
    ax_q.tick_params(axis="y", labelcolor=c_q)
    ax_q.spines["left"].set_edgecolor(c_q)

    ax_t.set_ylabel("TQUDO virt.", color=c_t)
    ax_t.tick_params(axis="y", labelcolor=c_t)
    ax_t.spines["right"].set_edgecolor(c_t)
    ax_t.spines["right"].set_visible(True)
    h1, lab1 = ax_q.get_legend_handles_labels()
    h2, lab2 = ax_t.get_legend_handles_labels()
    ax_q.legend(h1 + h2, lab1 + lab2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(images_dir / "energy_history_mean_cudaq_qubo_vs_tqudo_virtual_n5.png", dpi=150)
    plt.close(fig)

    # --- 2) Cirq TQUDO vs CUDA-Q TQUDO virtual, n=5; one shared TQUDO ref ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (solver, formulation, label) in enumerate(
        [
            ("cirq", "tqudo", "Cirq TQUDO"),
            ("cudaq", "tqudo_virtual", "CQ virt."),
        ]
    ):
        c = colors[i % len(colors)]
        df = _mean_energy_curve_by_step(
            curves, solver=solver, formulation=formulation, n_cities=5
        )
        _plot_mean_energy_with_std_band(ax, df, color=c, label=label)
    ref_shared = _mean_ref_tqudo_shared_n(paired, n_cities=5)
    if ref_shared is not None:
        ax.axhline(
            ref_shared,
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            label="BF optimum",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$f$ (mean ± σ)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(images_dir / "energy_history_mean_cirq_tqudo_vs_cudaq_tvirt_n5.png", dpi=150)
    plt.close(fig)

    # --- 3) Cirq TQUDO: each n_cities with matching ref ---
    cirq_t = curves[(curves["solver"] == "cirq") & (curves["formulation"] == "tqudo")]
    if cirq_t.empty:
        return
    n_list = sorted({int(x) for x in cirq_t["n_cities"].dropna().tolist()})
    if not n_list:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, n in enumerate(n_list):
        n_int = int(n)
        c = colors[i % len(colors)]
        df = _mean_energy_curve_by_step(
            curves, solver="cirq", formulation="tqudo", n_cities=n_int
        )
        if df.empty:
            continue
        _plot_mean_energy_with_std_band(
            ax, df, color=c, label=f"n = {n_int}"
        )
        ref = _mean_ref_objective(paired, solver="cirq", formulation="tqudo", n_cities=n_int)
        if ref is not None:
            ax.axhline(ref, color=c, linestyle="--", linewidth=1.2, alpha=0.88)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$f$ (mean ± σ)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(images_dir / "energy_history_mean_cirq_tqudo_by_ncities.png", dpi=150)
    plt.close(fig)
