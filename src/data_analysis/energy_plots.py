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


def _energy_curve_figsize(stem: str) -> tuple[float, float]:
    if stem.startswith("cudaq_qubo_tvirt_n5"):
        return (8.0, 4.5)
    return (9.0, 5.0)


def write_energy_history_plot_tables(paired: Any, curves: Any, plots_data_energy: Path) -> None:
    """Write one Parquet per energy-history figure (series rows: step, mean, std, ref_hline_y)."""
    import pandas as pd

    if curves is None or getattr(curves, "empty", True):
        return

    plots_data_energy.mkdir(parents=True, exist_ok=True)
    for stale in plots_data_energy.glob("cirq_tqudo_vs_cq_tvirt_n5_p*.parquet"):
        if stale.is_file():
            stale.unlink()

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

    for depth in depths_cudaq_n5:
        rows: list[dict[str, float | int | str]] = []
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
        if df_q.empty and df_t.empty:
            continue
        yq = _optimum_hline_y(paired, solver="cudaq", formulation="qubo", n_cities=5)
        yt = _optimum_hline_y(paired, solver="cudaq", formulation="tqudo_virtual", n_cities=5)
        h_q = float(yq) if yq is not None else float("nan")
        h_t = float(yt) if yt is not None else float("nan")
        if not df_q.empty:
            for _, r in df_q.iterrows():
                rows.append(
                    {
                        "series_label": "QUBO",
                        "step": int(r["step"]),
                        "mean": float(r["mean"]),
                        "std": float(r["std"]),
                        "ref_hline_y": h_q,
                    }
                )
        if not df_t.empty:
            for _, r in df_t.iterrows():
                rows.append(
                    {
                        "series_label": "TQUDO qubits",
                        "step": int(r["step"]),
                        "mean": float(r["mean"]),
                        "std": float(r["std"]),
                        "ref_hline_y": h_t,
                    }
                )
        pd.DataFrame(rows).to_parquet(
            plots_data_energy / f"cudaq_qubo_tvirt_n5_p{depth}.parquet",
            index=False,
        )

    series_cq_cirq: tuple[tuple[str, str, str], ...] = (
        ("cirq", "tqudo", "TQUDO qudits"),
        ("cudaq", "tqudo_virtual", "TQUDO qubits"),
    )
    for depth in sorted(depths_cirq_cudaq):
        rows = []
        plotted = False
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
                lab = f"{base_label}, n = {n_cc}"
                yn = _optimum_hline_y(
                    paired, solver=solver, formulation=formulation, n_cities=n_cc
                )
                hy = float(yn) if yn is not None else float("nan")
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "series_label": lab,
                            "step": int(r["step"]),
                            "mean": float(r["mean"]),
                            "std": float(r["std"]),
                            "ref_hline_y": hy,
                        }
                    )
                plotted = True
        if plotted:
            pd.DataFrame(rows).to_parquet(
                plots_data_energy / f"cirq_tqudo_vs_cq_tvirt_n5_n9_p{depth}.parquet",
                index=False,
            )

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
        rows = []
        any_line = False
        for n_int in n_list:
            df = _mean_energy_curve_by_step(
                curves,
                solver="cirq",
                formulation="tqudo",
                n_cities=n_int,
                qaoa_depth=depth,
            )
            if df.empty:
                continue
            lab = f"n = {n_int}"
            yn = _optimum_hline_y(
                paired, solver="cirq", formulation="tqudo", n_cities=n_int
            )
            hy = float(yn) if yn is not None else float("nan")
            for _, r in df.iterrows():
                rows.append(
                    {
                        "series_label": lab,
                        "step": int(r["step"]),
                        "mean": float(r["mean"]),
                        "std": float(r["std"]),
                        "ref_hline_y": hy,
                    }
                )
            any_line = True
        if any_line:
            pd.DataFrame(rows).to_parquet(
                plots_data_energy / f"cirq_tqudo_by_n_p{depth}.parquet",
                index=False,
            )


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


def run_energy_history_figures_from_disk(
    plots_data_energy: Path,
    images_energy: Path,
) -> None:
    """Render mean energy curves from ``plots_data/energy_history/*.parquet``."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if not plots_data_energy.is_dir():
        return

    images_energy.mkdir(parents=True, exist_ok=True)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    y_label_norm = r"$f\,/\,|f^*|$ (mean ± $\sigma$)"

    for stale in images_energy.glob("cirq_tqudo_vs_cq_tvirt_n5_p*.png"):
        if stale.is_file():
            stale.unlink()

    for pq in sorted(plots_data_energy.glob("*.parquet")):
        df = pd.read_parquet(pq)
        if df.empty:
            continue
        stem = pq.stem
        w, h = _energy_curve_figsize(stem)
        fig, ax = plt.subplots(figsize=(w, h))
        labels: list[str] = []
        seen: set[str] = set()
        for lab in df["series_label"].astype(str).tolist():
            if lab not in seen:
                seen.add(lab)
                labels.append(lab)
        for i, lab in enumerate(labels):
            sub = df.loc[df["series_label"] == lab, ["step", "mean", "std", "ref_hline_y"]].sort_values(
                "step"
            )
            c = colors[i % len(colors)]
            _plot_mean_energy_with_std_band(ax, sub, color=c, label=lab)
            hy = float(sub["ref_hline_y"].iloc[0])
            if np.isfinite(hy):
                ax.axhline(hy, color=c, linestyle="--", linewidth=1.2, alpha=0.88)
        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(y_label_norm, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        leg_fs = LEGEND_FONTSIZE_COMPACT if len(labels) > 3 else LEGEND_FONTSIZE
        ax.legend(fontsize=leg_fs, loc="best")
        fig.tight_layout()
        fig.savefig(images_energy / f"{stem}.png", dpi=150)
        plt.close(fig)
