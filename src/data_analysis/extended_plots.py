"""Figures for instance descriptors, efficiency, and QAOA angle similarity (reads ``processed/`` Parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_analysis._plot_typography import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_FONTSIZE_COMPACT,
    TICK_LABEL_FONTSIZE,
)


def _cohort_label(solver: str, formulation: str) -> str:
    if solver == "cudaq" and formulation == "qubo":
        return "CUDA-Q QUBO"
    if solver == "cudaq" and formulation == "tqudo_virtual":
        return "CUDA-Q V-QAOA"
    if solver == "cirq" and formulation == "tqudo":
        return "Cirq N-QAOA"
    return f"{solver} / {formulation}"


def _load_parquet_first(processed: Path, stem: str) -> Any:
    import pandas as pd

    pq = processed / f"{stem}.parquet"
    if pq.is_file():
        return pd.read_parquet(pq)
    csv = processed / f"{stem}.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return None


def run_extended_analysis_figures(processed: Path, images_extended: Path) -> None:
    """Write PNGs under *images_extended* from tables in *processed* (skips missing inputs)."""
    import matplotlib.pyplot as plt

    processed = processed.resolve()
    images_extended.mkdir(parents=True, exist_ok=True)

    paired = _load_parquet_first(processed, "paired_metrics")
    summary = _load_parquet_first(processed, "summary_by_config")
    angle_cohort = _load_parquet_first(processed, "angle_cohort_stats")
    paired_cq_ci = _load_parquet_first(processed, "paired_angle_cudaq_tvirt_cirq_tqudo")
    paired_q_t = _load_parquet_first(processed, "paired_angle_cudaq_qubo_tvirt")

    if paired is not None:
        _fig_instance_precedence_vs_rho(
            paired, images_extended / "instance_precedence_density_vs_rho.png"
        )
        _fig_efficiency_runtime_vs_rho(paired, images_extended / "efficiency_runtime_vs_rho.png")
        _fig_efficiency_configs_vs_rho(
            paired, images_extended / "efficiency_configs_evaluated_vs_rho.png"
        )

    if summary is not None:
        _fig_efficiency_runtime_by_depth(
            summary, images_extended / "efficiency_mean_runtime_by_depth.png"
        )

    if angle_cohort is not None:
        _fig_angle_cohort_pairwise_cosine(
            angle_cohort, images_extended / "angles_mean_pairwise_cosine_by_depth.png"
        )

    if paired_cq_ci is not None and not paired_cq_ci.empty:
        _fig_angle_pair_histogram(
            paired_cq_ci,
            "cosine_pair",
            images_extended / "angles_cudaq_virt_cirq_cosine_hist.png",
            title="CUDA-Q V-QAOA vs Cirq N-QAOA: angle similarity",
            xlabel="Cosine similarity of normalized $(\\gamma, \\beta)$ vectors",
            caption="Per instance: inner join on $(n,\\mathrm{inst},p)$. 1 = identical direction.",
        )
        _fig_angle_pair_histogram(
            paired_cq_ci,
            "l2_delta",
            images_extended / "angles_cudaq_virt_cirq_l2_hist.png",
            title="CUDA-Q V-QAOA vs Cirq N-QAOA: $\\|\\mathbf{v}_{\\mathrm{CQ}}-\\mathbf{v}_{\\mathrm{Ci}}\\|_2$",
            xlabel="L2 distance (both vectors L2-normalized)",
            caption="0 = identical angles. Larger = more different optimal QAOA parameters.",
        )

    if paired_q_t is not None and not paired_q_t.empty:
        _fig_angle_pair_histogram(
            paired_q_t,
            "cosine_pair",
            images_extended / "angles_cudaq_qubo_virt_cosine_hist.png",
            title="CUDA-Q QUBO vs V-QAOA: angle similarity",
            xlabel="Cosine similarity of normalized $(\\gamma, \\beta)$ vectors",
            caption="Same instance and depth $p$ on both formulations.",
        )

    plt.close("all")


def _fig_instance_precedence_vs_rho(paired: Any, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    need = {"inst_precedence_density", "approx_ratio_real", "solver", "formulation"}
    if not need.issubset(paired.columns):
        return
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & paired["solver"].ne("brute_force")
        & paired["solver"].ne("simulated_annealing")
    )
    sub = paired.loc[m].copy()
    sub = sub[sub["approx_ratio_real"].notna() & np.isfinite(sub["approx_ratio_real"])]
    sub = sub[sub["inst_precedence_density"].notna() & np.isfinite(sub["inst_precedence_density"])]
    if sub.empty:
        return
    if "feasible" in sub.columns:
        sub = sub[sub["feasible"].fillna(False)]

    sub["lab"] = sub.apply(
        lambda r: _cohort_label(str(r["solver"]), str(r["formulation"])),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, sub["lab"].nunique())))
    for i, lab in enumerate(sorted(sub["lab"].unique())):
        chunk = sub[sub["lab"] == lab]
        ax.scatter(
            chunk["inst_precedence_density"],
            chunk["approx_ratio_real"],
            alpha=0.35,
            s=22,
            label=lab,
            color=colors[i % len(colors)],
            edgecolors="none",
        )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, label=r"Optimal $\rho=1$")
    ax.set_xlabel(r"Precedence density (# precedences / $(n-1)^2$)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Approx. ratio $\rho$ (real cost / BF optimum)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.92)
    ax.set_title(
        "Instance constraints vs solution quality (feasible QAOA runs)",
        fontsize=AXIS_LABEL_FONTSIZE + 1,
    )
    fig.text(
        0.5,
        0.01,
        "Each point is one solution JSON. For regression / partial correlations use paired_metrics.parquet.",
        ha="center",
        fontsize=11,
        color="0.35",
    )
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_efficiency_runtime_vs_rho(paired: Any, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    need = {"runtime_seconds", "approx_ratio_real", "solver", "formulation"}
    if not need.issubset(paired.columns):
        return
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & paired["solver"].ne("brute_force")
        & paired["solver"].ne("simulated_annealing")
    )
    sub = paired.loc[m].copy()
    sub = sub[sub["approx_ratio_real"].notna() & np.isfinite(sub["approx_ratio_real"])]
    sub = sub[sub["runtime_seconds"].notna() & (sub["runtime_seconds"] > 0)]
    if sub.empty:
        return
    if "feasible" in sub.columns:
        sub = sub[sub["feasible"].fillna(False)]

    sub["lab"] = sub.apply(
        lambda r: _cohort_label(str(r["solver"]), str(r["formulation"])),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, sub["lab"].nunique())))
    for i, lab in enumerate(sorted(sub["lab"].unique())):
        chunk = sub[sub["lab"] == lab]
        ax.scatter(
            chunk["runtime_seconds"],
            chunk["approx_ratio_real"],
            alpha=0.35,
            s=24,
            label=lab,
            color=colors[i % len(colors)],
            edgecolors="none",
        )
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Runtime (s), log scale", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Approx. ratio $\rho$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.92)
    ax.set_title("Quality vs wall-clock time", fontsize=AXIS_LABEL_FONTSIZE + 1)
    fig.text(
        0.5,
        0.01,
        "configs_evaluated and runtime_per_energy_step are in paired_metrics / summary_by_config (CSV).",
        ha="center",
        fontsize=11,
        color="0.35",
    )
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_efficiency_configs_vs_rho(paired: Any, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    base_need = {"approx_ratio_real", "solver", "formulation"}
    if not base_need.issubset(paired.columns):
        return
    m = (
        paired["parse_ok"]
        & paired["solve_ok"]
        & paired["solver"].ne("brute_force")
        & paired["solver"].ne("simulated_annealing")
    )
    base = paired.loc[m].copy()
    base = base[base["approx_ratio_real"].notna() & np.isfinite(base["approx_ratio_real"])]
    if base.empty:
        return
    if "feasible" in base.columns:
        base = base[base["feasible"].fillna(False)]

    sub: Any | None = None
    x_col = "configs_evaluated"
    if "configs_evaluated" in base.columns:
        t = base[base["configs_evaluated"].notna()].copy()
        if not t.empty:
            t["configs_evaluated"] = t["configs_evaluated"].astype(float)
            t = t[np.isfinite(t["configs_evaluated"]) & (t["configs_evaluated"] > 0)]
            if not t.empty:
                sub = t

    if sub is None:
        if "n_energy_steps" not in base.columns:
            return
        t = base[base["n_energy_steps"].notna()].copy()
        t["n_energy_steps"] = t["n_energy_steps"].astype(float)
        t = t[np.isfinite(t["n_energy_steps"]) & (t["n_energy_steps"] > 0)]
        if t.empty:
            return
        sub = t
        x_col = "n_energy_steps"
    if sub is None or sub.empty:
        return

    sub["lab"] = sub.apply(
        lambda r: _cohort_label(str(r["solver"]), str(r["formulation"])),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, sub["lab"].nunique())))
    for i, lab in enumerate(sorted(sub["lab"].unique())):
        chunk = sub[sub["lab"] == lab]
        ax.scatter(
            chunk[x_col],
            chunk["approx_ratio_real"],
            alpha=0.35,
            s=24,
            label=lab,
            color=colors[i % len(colors)],
            edgecolors="none",
        )
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    if x_col == "configs_evaluated":
        x_label = "configs_evaluated (log scale)"
        sub_title = "Optimizer work vs solution quality"
        foot = (
            "configs_evaluated: filled mainly by brute_force in JSON; "
            "QAOA backends often omit it — then this plot would be empty without a fallback."
        )
    else:
        x_label = r"Energy evaluations ($\mathit{len}(\mathrm{energy\_history})$), log scale"
        sub_title = "Classical QAOA loop length vs solution quality (proxy)"
        foot = (
            "QAOA JSON usually has no configs_evaluated; using n_energy_steps = "
            "optimizer steps recorded in energy_history."
        )
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Approx. ratio $\rho$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.92)
    ax.set_title(sub_title, fontsize=AXIS_LABEL_FONTSIZE + 1)
    fig.text(0.5, 0.01, foot, ha="center", fontsize=11, color="0.35")
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_efficiency_runtime_by_depth(summary: Any, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    need = {"n_cities", "solver", "formulation", "qaoa_depth", "mean_runtime"}
    if not need.issubset(summary.columns):
        return
    sub = summary.dropna(subset=["qaoa_depth", "mean_runtime"]).copy()
    sub = sub[np.isfinite(sub["mean_runtime"]) & (sub["mean_runtime"] > 0)]
    if sub.empty:
        return
    sub["qaoa_depth"] = sub["qaoa_depth"].astype(int)
    sub["lab"] = sub.apply(
        lambda r: (
            f"n={int(r['n_cities'])} · {_cohort_label(str(r['solver']), str(r['formulation']))}"
        ),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in sorted(sub["lab"].unique()):
        chunk = sub[sub["lab"] == lab].sort_values("qaoa_depth")
        ax.plot(
            chunk["qaoa_depth"],
            chunk["mean_runtime"],
            marker="o",
            linewidth=2,
            markersize=7,
            label=lab,
        )
    ax.set_xlabel(r"QAOA depth $p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mean runtime (s) per group", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(sorted(sub["qaoa_depth"].unique()))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE_COMPACT, framealpha=0.92)
    ax.set_title(
        "Computational cost by configuration (summary_by_config)", fontsize=AXIS_LABEL_FONTSIZE + 1
    )
    fig.text(
        0.5,
        0.01,
        "Median runtime: median_runtime_seconds. Cost per optimizer step: mean_runtime_per_energy_step.",
        ha="center",
        fontsize=11,
        color="0.35",
    )
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_angle_cohort_pairwise_cosine(angle_cohort: Any, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    need = {"n_cities", "solver", "formulation", "qaoa_depth", "mean_pairwise_cosine"}
    if not need.issubset(angle_cohort.columns):
        return
    sub = angle_cohort.copy()
    if "n_runs_angles_dim_consistent" in sub.columns:
        sub = sub[sub["n_runs_angles_dim_consistent"].fillna(0) >= 2]
    sub = sub[sub["mean_pairwise_cosine"].notna() & np.isfinite(sub["mean_pairwise_cosine"])]
    sub = sub.dropna(subset=["qaoa_depth"])
    if sub.empty:
        return
    sub["qaoa_depth"] = sub["qaoa_depth"].astype(int)
    sub["lab"] = sub.apply(
        lambda r: (
            f"n={int(r['n_cities'])} · {_cohort_label(str(r['solver']), str(r['formulation']))}"
        ),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in sorted(sub["lab"].unique()):
        chunk = sub[sub["lab"] == lab].sort_values("qaoa_depth")
        ax.plot(
            chunk["qaoa_depth"],
            chunk["mean_pairwise_cosine"],
            marker="s",
            linewidth=2,
            markersize=7,
            label=lab,
        )
    ax.axhline(1.0, color="0.65", linestyle=":", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mean pairwise cosine (instances in cohort)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(sorted(sub["qaoa_depth"].unique()))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE_COMPACT, framealpha=0.92)
    ax.set_title(
        "Are optimal QAOA angles similar across instances? (higher → more reusable)",
        fontsize=AXIS_LABEL_FONTSIZE + 1,
    )
    fig.text(
        0.5,
        0.01,
        "Built from angle_cohort_stats: only runs sharing the same vector length (see n_runs_angles_dim_consistent).",
        ha="center",
        fontsize=11,
        color="0.35",
    )
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_angle_pair_histogram(
    df: Any,
    col: str,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    caption: str,
) -> None:
    import matplotlib.pyplot as plt

    if col not in df.columns:
        return
    vals = pd_to_numpy_finite(df[col])
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        vals, bins=min(40, max(10, vals.size // 5)), color="#4c72b0", edgecolor="white", alpha=0.9
    )
    ax.axvline(
        float(np.mean(vals)),
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"mean = {np.mean(vals):.3f}",
    )
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE + 1)
    fig.text(0.5, 0.02, caption, ha="center", fontsize=11, color="0.35", style="italic")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pd_to_numpy_finite(s: Any) -> np.ndarray:
    import pandas as pd

    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return x[np.isfinite(x)]
