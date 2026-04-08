"""Matplotlib figures for CUDA-Q/Cirq benchmark dashboards and boxplots."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from data_analysis._plot_typography import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_FONTSIZE_COMPACT,
    TICK_LABEL_FONTSIZE,
)
from data_analysis.benchmark.common import (
    _LOG_Y_FLOOR,
    _P_OPT_LOG_AXIS_FLOOR,
    _clip_values_for_log_y,
    _uniform_superposition_p_opt_htsp,
    _uniform_superposition_p_opt_qubo,
)


def _dashboard_bar_count(
    row: dict[str, float | int],
    *,
    count_key: str,
    pct_key: str,
    denom_key: str,
) -> float:
    """Integer instance count for dashboard bars; supports legacy rows with only %% and *denom*."""
    raw = row.get(count_key)
    if raw is not None:
        try:
            v = float(raw)
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    pct = float(row.get(pct_key, float("nan")))
    d_raw = row.get(denom_key)
    if d_raw is None:
        return 0.0
    d = int(d_raw)
    if d <= 0 or not math.isfinite(pct):
        return 0.0
    return float(round(pct * float(d) / 100.0))


def _plot_comparison_dashboard(
    *,
    x_labels: list[str],
    stats_list: list[dict[str, float | int]],
    label_left: str,
    label_right: str,
    x_axis_label: str,
    other_panels_stats_stop: int | None = None,
) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    if other_panels_stats_stop is not None:
        stop = int(other_panels_stats_stop)
        stats_rest = stats_list[:stop]
        x_labels_rest = x_labels[:stop]
        if not stats_rest or len(stats_rest) != len(x_labels_rest):
            raise ValueError(
                "other_panels_stats_stop must slice stats_list and x_labels to the same positive length"
            )
    else:
        stats_rest = stats_list
        x_labels_rest = x_labels

    x = np.arange(len(x_labels), dtype=np.float64)
    x_rest = np.arange(len(x_labels_rest), dtype=np.float64)
    w = 0.36
    c_opt, c_sub, c_inf = "#2ca02c", "#ffbb78", "#c7c7c7"

    ax00 = axes[0, 0]
    left_opt = [float(s["left_optimal"]) for s in stats_list]
    left_sub = [float(s["left_feasible_subopt"]) for s in stats_list]
    left_inf = [float(s["left_infeasible"]) for s in stats_list]
    right_opt = [float(s["right_optimal"]) for s in stats_list]
    right_sub = [float(s["right_feasible_subopt"]) for s in stats_list]
    right_inf = [float(s["right_infeasible"]) for s in stats_list]

    ax00.bar(x - w / 2, left_opt, w, label="Optimal", color=c_opt, edgecolor="white", linewidth=0.5)
    ax00.bar(
        x - w / 2,
        left_sub,
        w,
        bottom=left_opt,
        label="Suboptimal",
        color=c_sub,
        edgecolor="white",
        linewidth=0.5,
    )
    ax00.bar(
        x - w / 2,
        left_inf,
        w,
        bottom=np.array(left_opt) + np.array(left_sub),
        label="Infeasible",
        color=c_inf,
        edgecolor="white",
        linewidth=0.5,
    )

    ax00.bar(x + w / 2, right_opt, w, color=c_opt, edgecolor="white", linewidth=0.5)
    ax00.bar(
        x + w / 2,
        right_sub,
        w,
        bottom=right_opt,
        color=c_sub,
        edgecolor="white",
        linewidth=0.5,
    )
    ax00.bar(
        x + w / 2,
        right_inf,
        w,
        bottom=np.array(right_opt) + np.array(right_sub),
        color=c_inf,
        edgecolor="white",
        linewidth=0.5,
    )

    ax00.set_xticks(x)
    ax00.set_xticklabels(x_labels)
    ax00.set_ylabel("Instances", fontsize=AXIS_LABEL_FONTSIZE)
    ax00.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax00.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax00.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax01 = axes[0, 1]
    cl = [
        _dashboard_bar_count(
            s,
            count_key="cost_left_better_cond",
            pct_key="cost_left_better_cond_pct",
            denom_key="n_both_feasible",
        )
        for s in stats_rest
    ]
    cr = [
        _dashboard_bar_count(
            s,
            count_key="cost_right_better_cond",
            pct_key="cost_right_better_cond_pct",
            denom_key="n_both_feasible",
        )
        for s in stats_rest
    ]
    ct = [
        _dashboard_bar_count(
            s,
            count_key="cost_tie_cond",
            pct_key="cost_tie_cond_pct",
            denom_key="n_both_feasible",
        )
        for s in stats_rest
    ]
    ww = 0.25
    ax01.bar(x_rest - ww, cl, ww, label=f"Lower cost ({label_left})", color="#1f77b4")
    ax01.bar(x_rest, cr, ww, label=f"Lower cost ({label_right})", color="#ff7f0e")
    ax01.bar(x_rest + ww, ct, ww, label="Tie", color="#7f7f7f")
    ax01.set_xticks(x_rest)
    ax01.set_xticklabels(x_labels_rest)
    ax01.set_ylabel(
        "Instances\n(both feasible, lower real cost / tie)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax01.set_ylim(bottom=0)
    ax01.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=4))
    ax01.legend(fontsize=LEGEND_FONTSIZE_COMPACT)
    ax01.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax01.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax10 = axes[1, 0]
    olf = [
        _dashboard_bar_count(
            s,
            count_key="only_left_feasible",
            pct_key="only_left_feasible_pct",
            denom_key="n_paired",
        )
        for s in stats_rest
    ]
    orf = [
        _dashboard_bar_count(
            s,
            count_key="only_right_feasible",
            pct_key="only_right_feasible_pct",
            denom_key="n_paired",
        )
        for s in stats_rest
    ]
    ax10.bar(x_rest - ww, olf, ww, label=f"{label_left} only", color="#1f77b4")
    ax10.bar(x_rest + ww, orf, ww, label=f"{label_right} only", color="#ff7f0e")
    ax10.set_xticks(x_rest)
    ax10.set_xticklabels(x_labels_rest)
    ax10.set_ylabel(
        "Instances\n(feasible on one side only)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax10.set_ylim(bottom=0)
    ax10.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=4))
    ax10.legend(fontsize=LEGEND_FONTSIZE)
    ax10.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax10.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    ax11 = axes[1, 1]
    olo = [
        _dashboard_bar_count(
            s,
            count_key="only_left_optimal",
            pct_key="only_left_optimal_pct",
            denom_key="n_paired",
        )
        for s in stats_rest
    ]
    oro = [
        _dashboard_bar_count(
            s,
            count_key="only_right_optimal",
            pct_key="only_right_optimal_pct",
            denom_key="n_paired",
        )
        for s in stats_rest
    ]
    ax11.bar(x_rest - ww, olo, ww, label=f"{label_left} only", color="#2ca02c")
    ax11.bar(x_rest + ww, oro, ww, label=f"{label_right} only", color="#d62728")
    ax11.set_xticks(x_rest)
    ax11.set_xticklabels(x_labels_rest)
    ax11.set_ylabel(
        "Instances\n(optimal on one side only)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax11.set_ylim(bottom=0)
    ax11.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=4))
    ax11.legend(fontsize=LEGEND_FONTSIZE)
    ax11.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax11.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    return fig


# Strip / jitter under box patches so boxes read on top (matplotlib paints higher z-order later).
_ZORDER_STRIP_SCATTER_POINTS = 1.0
_ZORDER_BOX_ARTISTS = 3.0


def _style_boxplot_patches_and_zorder(
    bp: dict[str, Any],
    *,
    facecolor: str,
    face_alpha: float,
    z_box: float = _ZORDER_BOX_ARTISTS,
) -> None:
    """Raise all boxplot artists above strip scatters; color box faces only."""
    for artists in bp.values():
        for artist in artists:
            artist.set_zorder(z_box)
    for patch in bp["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(face_alpha)


def _scatter_rho_instances_jittered(
    ax: Any,
    x_center: float,
    values: list[float],
    *,
    color: str,
    jitter_span: float,
    rng: np.random.Generator,
) -> None:
    """Jittered semi-transparent markers so stacked :math:`\\rho=1` instances remain visible."""
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    j = (rng.random(arr.size) - 0.5) * jitter_span
    ax.scatter(
        x_center + j,
        arr,
        s=18,
        c=color,
        alpha=0.32,
        edgecolors="white",
        linewidths=0.25,
        zorder=_ZORDER_STRIP_SCATTER_POINTS,
    )


def _ylim_rho_plot(all_values: list[float], *, y_scale: str) -> tuple[float, float]:
    if not all_values:
        return (1.0, 1.05) if y_scale == "log" else (1.0, 1.05)
    arr = np.asarray(all_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (1.0, 1.05) if y_scale == "log" else (1.0, 1.05)
    mx = float(np.max(arr))
    if y_scale == "log":
        return 1.0, max(mx * 1.12, 1.02)
    mn = float(np.min(arr))
    bottom = 1.0 if mn >= 1.0 else max(0.95, mn * 0.98)
    pad = max((mx - 1.0) * 0.1, 0.04) if mx > 1.0 else 0.05
    return bottom, max(mx + pad, bottom + 0.04)


def _decorate_approx_ratio_y_axis(
    ax: Any,
    lo: float,
    hi: float,
    *,
    y_scale: str,
    symlog_linthresh: float,
) -> None:
    """Readable ticks for :math:`\\rho \\gtrsim 1` (linear default; log uses per-decade multipliers)."""
    from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, ScalarFormatter

    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    elif y_scale == "log":
        ax.set_yscale("log")
        subs = tuple(float(x) for x in range(1, 10))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=15))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.set_yscale("linear")
        span = max(hi - lo, 1e-9)
        if span <= 0.2:
            step = 0.02 if span <= 0.1 else 0.05
            ax.yaxis.set_major_locator(MultipleLocator(step))
        elif span <= 0.5:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
        else:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10], min_n_ticks=5)
            )
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)

    ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.35)
    if y_scale == "log":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.15)


def _decorate_y_axis_from_values(
    ax: Any,
    all_y: list[float],
    *,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    asinh_linear_width: float | None = None,
    log_y_clip_upper: float | None = None,
    manual_y_limits: bool = True,
) -> None:
    """Y limits with padding and readable ticks for generic numeric samples.

    ``y_scale="asinh"`` uses Matplotlib's inverse-sinh axis (often clearer ticks
    than ``symlog`` for signed values). When ``asinh_linear_width`` is omitted,
    ``symlog_linthresh`` is used as the ``linear_width`` parameter.

    For ``y_scale="log"``, if ``log_y_clip_upper`` is set (e.g. ``1.0`` for
    :math:`P(\\mathrm{opt})`), the upper *y* limit is capped at that value.

    If ``manual_y_limits`` is false, no ``set_ylim`` is applied; the axis uses
    Matplotlib autoscaling from artists already drawn on *ax*.
    """
    from matplotlib.ticker import (
        AsinhLocator,
        LogFormatterSciNotation,
        LogLocator,
        MaxNLocator,
        ScalarFormatter,
        SymmetricalLogLocator,
    )

    arr = np.asarray([v for v in all_y if np.isfinite(v)], dtype=np.float64)
    if y_scale == "log":
        pos = arr[arr > 0]
        if pos.size == 0:
            lo, hi = float(_P_OPT_LOG_AXIS_FLOOR), 1.0
        else:
            lo_raw = float(np.min(pos))
            hi = float(np.max(arr))
            hi = max(hi, lo_raw)
            # Padding below the smallest *y*; do not clamp to ``_P_OPT_LOG_AXIS_FLOOR``
            # (too high—hides uniform references like ``1/2^{(n-1)^2}`` for large *n*).
            lo = max(lo_raw * 0.62, float(np.finfo(float).tiny))
            hi = max(hi * 1.18, lo_raw * 1.06)
            if log_y_clip_upper is not None:
                hi = min(hi, float(log_y_clip_upper))
        if lo >= hi:
            hi = lo * 10.0
            if log_y_clip_upper is not None:
                hi = min(hi, float(log_y_clip_upper))
        ax.set_yscale("log")
        if manual_y_limits:
            ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    else:
        if arr.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(np.min(arr)), float(np.max(arr))
            span = hi - lo
            abs_max = float(np.max(np.abs(arr)))
            edge_max = max(abs(lo), abs(hi))
            if span > 0:
                pad = max(span * 0.18, 0.03 * edge_max, 0.02 * abs_max, 1e-9)
            else:
                pad = max(0.05 * edge_max, 0.04 * abs_max, 0.02, 1e-9)
            lo, hi = lo - pad, hi + pad
        if y_scale == "asinh":
            lw = (
                float(asinh_linear_width)
                if asinh_linear_width is not None
                else float(symlog_linthresh)
            )
            lw = max(lw, 1e-15)
            ax.set_yscale("asinh", linear_width=lw)
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(AsinhLocator(lw, numticks=8))
        elif y_scale == "symlog":
            absv = np.abs(arr[np.isfinite(arr)])
            absv_nz = absv[absv > 1e-15]
            if absv_nz.size:
                med = float(np.median(absv_nz))
                eff_lt = max(symlog_linthresh, min(med * 0.15, 0.05))
            else:
                eff_lt = max(symlog_linthresh, 1e-6)
            ax.set_yscale("symlog", linthresh=eff_lt, base=10)
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=eff_lt, base=10))
        else:
            ax.set_yscale("linear")
            if manual_y_limits:
                ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    if not manual_y_limits:
        ax.relim(visible_only=True)
        ax.autoscale(axis="y")
    ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.35)
    if y_scale == "log":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.2)
    elif y_scale == "symlog":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.22)
    elif y_scale == "asinh":
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.18)


def _plot_approx_ratio_boxplots_vs_p(
    series: list[tuple[str, dict[int, list[float]]]],
    *,
    figsize: tuple[float, float] = (7.5, 4.8),
    y_label: str | None = None,
    ref_hline: float | None = None,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    strip_jitter: bool = True,
    y_axis_kind: str = "rho",
    y_floor: float | None = None,
    log_y_clip_upper: float | None = None,
    uniform_p_opt_hline_ns: tuple[int, ...] = (),
    uniform_qubo_p_opt_hline_ns: tuple[int, ...] = (),
    uniform_refs_in_ylim: bool = True,
) -> Any:
    """Boxplots vs QAOA depth *p* (dodged by series).

    ``y_axis_kind="rho"``: limits/ticks for approximation ratio; ``"generic"`` uses
    :func:`_decorate_y_axis_from_values` (e.g. :math:`P(\\mathrm{opt})`).

    ``uniform_p_opt_hline_ns``: draw horizontal reference :math:`1/(n-1)^{n-1}` per :math:`n` (tensor / N-QAOA counting).
    ``uniform_qubo_p_opt_hline_ns``: draw horizontal reference :math:`1/2^{(n-1)^2}` per :math:`n` (QUBO).
    Uniform lines use the same face color as the **N-QAOA** and **QUBO** box series when those
    labels exist in *series*.

    If ``uniform_refs_in_ylim`` is false (log *y* only), reference masses are not mixed into *y* limits
    so the axis follows sample/box data (reference lines may fall outside the visible range).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]

    active = [(lab, dct) for lab, dct in series if dct]
    if not active:
        ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(
            y_label if y_label is not None else r"$\rho$",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        fig.tight_layout()
        return fig

    all_depths = sorted({d for _, dct in active for d in dct})
    n_s = len(active)
    r_tq = next(
        (
            r
            for r, (lab, _) in enumerate(active)
            if lab == "N-QAOA" or lab.startswith("N-QAOA")
        ),
        None,
    )
    r_qb = next(
        (r for r, (lab, _) in enumerate(active) if lab == "QUBO" or lab.startswith("QUBO")),
        None,
    )
    col_tqudo_uni = colors[(r_tq if r_tq is not None else 0) % len(colors)]
    col_qubo_uni = colors[(r_qb if r_qb is not None else max(n_s - 1, 0)) % len(colors)]
    if_narrow = max(n_s - 1, 1)
    if n_s <= 3:
        dodge = 0.09
    else:
        dodge = 0.43 / float(if_narrow)
    half = 0.5 * float(n_s - 1) if n_s > 1 else 0.0
    box_w_rel = 0.88
    jitter_w = min(dodge * box_w_rel * 0.88, 0.075)
    legend_handles: list[Any] = []
    all_rho: list[float] = []

    for rank, (label, dct) in enumerate(active):
        color = colors[rank % len(colors)]
        positions: list[float] = []
        col_data: list[list[float]] = []
        for d in all_depths:
            vals = dct.get(int(d), [])
            if not vals:
                continue
            if y_scale == "log" and y_axis_kind == "generic":
                vals = _clip_values_for_log_y(list(vals))
            positions.append(float(d) + (float(rank) - half) * dodge)
            col_data.append(vals)
            all_rho.extend(float(v) for v in vals if np.isfinite(v))
        if not col_data:
            continue
        if strip_jitter:
            for pos, vals in zip(positions, col_data, strict=True):
                seed = 9001 + rank * 131 + int(round(pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax,
                    pos,
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(seed),
                )
        bp = ax.boxplot(
            col_data,
            positions=positions,
            widths=dodge * box_w_rel,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        _style_boxplot_patches_and_zorder(bp, facecolor=color, face_alpha=0.45)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label))

    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(float(ref_hline), color="gray", linestyle="--", linewidth=1, zorder=1)
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label=rlab)
            )

    if uniform_refs_in_ylim:
        for un in uniform_p_opt_hline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(un))
            if np.isfinite(pu) and pu > 0.0:
                all_rho.append(float(pu))
        for un in uniform_qubo_p_opt_hline_ns:
            pq = _uniform_superposition_p_opt_qubo(int(un))
            if np.isfinite(pq) and pq > 0.0:
                all_rho.append(float(pq))

    if all_depths:
        ax.set_xticks([float(d) for d in all_depths])
        ax.set_xticklabels([str(d) for d in all_depths])
    ax.set_xlabel(r"$p$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if y_axis_kind == "rho":
        lo, hi = _ylim_rho_plot(all_rho, y_scale=y_scale)
        _decorate_approx_ratio_y_axis(
            ax, lo, hi, y_scale=y_scale, symlog_linthresh=symlog_linthresh
        )
        ax.set_ylim(lo, hi)
    else:
        _decorate_y_axis_from_values(
            ax,
            all_rho,
            y_scale=y_scale,
            symlog_linthresh=symlog_linthresh,
            log_y_clip_upper=log_y_clip_upper,
        )
        if y_floor is not None:
            y_lo = float(y_floor)
            if y_scale == "log" and y_lo <= 0.0:
                y_lo = float(_P_OPT_LOG_AXIS_FLOOR)
            _, cur_hi = ax.get_ylim()
            ax.set_ylim(y_lo, cur_hi)
    for un in uniform_p_opt_hline_ns:
        pu = _uniform_superposition_p_opt_htsp(int(un))
        if not (np.isfinite(pu) and pu > 0.0):
            continue
        ax.axhline(
            float(pu),
            color=col_tqudo_uni,
            linestyle="--",
            linewidth=1.25,
            alpha=0.9,
            zorder=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=col_tqudo_uni,
                linestyle="--",
                linewidth=1.25,
                label=rf"Uniform N-QAOA $n={un}$ ($1/{un - 1}^{{{un - 1}}}$)",
            )
        )
    for un in uniform_qubo_p_opt_hline_ns:
        pq = _uniform_superposition_p_opt_qubo(int(un))
        if not (np.isfinite(pq) and pq > 0.0):
            continue
        ax.axhline(
            float(pq),
            color=col_qubo_uni,
            linestyle=":",
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )
        exp = (int(un) - 1) ** 2
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=col_qubo_uni,
                linestyle=":",
                linewidth=1.4,
                label=rf"Uniform QUBO $n={un}$ ($1/2^{{{exp}}}$)",
            )
        )
    if (
        uniform_refs_in_ylim
        and y_axis_kind == "generic"
        and y_scale == "log"
        and (uniform_p_opt_hline_ns or uniform_qubo_p_opt_hline_ns)
    ):
        ref_lo: list[float] = []
        for un in uniform_p_opt_hline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(un))
            if np.isfinite(pu) and pu > 0.0:
                ref_lo.append(float(pu))
        for un in uniform_qubo_p_opt_hline_ns:
            pq = _uniform_superposition_p_opt_qubo(int(un))
            if np.isfinite(pq) and pq > 0.0:
                ref_lo.append(float(pq))
        if ref_lo:
            y_a, y_b = ax.get_ylim()
            ax.set_ylim(min(y_a, min(ref_lo) * 0.35), y_b)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_approx_ratio_boxplots_vs_ncities(
    series: list[tuple[str, list[float], list[list[float]]]],
    *,
    n_tick_vals: list[int],
    figsize: tuple[float, float] = (8.0, 5.0),
    y_label: str | None = None,
    ref_hline: float | None = None,
    ref_hline_label: str | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    strip_jitter: bool = True,
) -> Any:
    """``series``: (label, x positions, one list of ρ per box, aligned with *x*)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    legend_handles: list[Any] = []
    box_w = 0.1
    jitter_w = min(box_w * 0.78, 0.07)
    all_rho: list[float] = []

    for i, (label, xs, datas) in enumerate(series):
        if not xs or not datas or len(xs) != len(datas):
            continue
        color = colors[i % len(colors)]
        for vals in datas:
            all_rho.extend(float(v) for v in vals if np.isfinite(v))
        if strip_jitter:
            for x_pos, vals in zip(xs, datas, strict=True):
                seed = 4242 + i * 97 + int(round(x_pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax,
                    float(x_pos),
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(seed),
                )
        bp = ax.boxplot(
            datas,
            positions=xs,
            widths=box_w,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        _style_boxplot_patches_and_zorder(bp, facecolor=color, face_alpha=0.45)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label))

    if ref_hline is not None and math.isfinite(float(ref_hline)):
        if y_scale != "log" or float(ref_hline) > 0.0:
            rlab = ref_hline_label if ref_hline_label is not None else r"$\rho = 1$"
            ax.axhline(float(ref_hline), color="gray", linestyle="--", linewidth=1, zorder=1)
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label=rlab)
            )

    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(r"$n$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        y_label if y_label is not None else r"$\rho$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    lo, hi = _ylim_rho_plot(all_rho, y_scale=y_scale)
    _decorate_approx_ratio_y_axis(ax, lo, hi, y_scale=y_scale, symlog_linthresh=symlog_linthresh)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_dodged_boxplot_series_vs_ncities(
    series: list[tuple[str, list[float], list[list[float]]]],
    *,
    n_tick_vals: list[int],
    xlabel: str = r"$n$",
    y_label: str,
    figsize: tuple[float, float] = (8.5, 5.0),
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
    asinh_linear_width: float | None = None,
    strip_jitter: bool = True,
    y_floor: float | None = None,
    log_y_clip_upper: float | None = None,
    manual_y_limits: bool = True,
    uniform_p_opt_vline_ns: list[int] | None = None,
) -> Any:
    """Dodged box/strip series vs city count (same layout as ρ vs *n*; generic *y* axis).

    If ``uniform_p_opt_vline_ns`` is set, draw a dashed polyline through each
    reference point :math:`(n, 1/(n-1)^{n-1})` (markers at every :math:`n` so all stay
    visible on log *y*).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    legend_handles: list[Any] = []
    box_w = 0.1
    jitter_w = min(box_w * 0.78, 0.07)
    all_y: list[float] = []

    for i, (label, xs, datas) in enumerate(series):
        if not xs or not datas or len(xs) != len(datas):
            continue
        color = colors[i % len(colors)]
        draw_lists = [_clip_values_for_log_y(list(v)) for v in datas] if y_scale == "log" else datas
        for vals in draw_lists:
            all_y.extend(float(v) for v in vals if np.isfinite(v))
        if strip_jitter:
            for x_pos, vals in zip(xs, draw_lists, strict=True):
                seed = 8080 + i * 97 + int(round(x_pos * 1000)) + len(vals)
                _scatter_rho_instances_jittered(
                    ax,
                    float(x_pos),
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(seed),
                )
        bp = ax.boxplot(
            draw_lists,
            positions=xs,
            widths=box_w,
            patch_artist=True,
            showfliers=not strip_jitter,
            boxprops=dict(linewidth=1.0, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
            flierprops=dict(
                marker="o",
                markerfacecolor=color,
                markersize=3,
                alpha=0.45,
                linestyle="none",
            ),
        )
        _style_boxplot_patches_and_zorder(bp, facecolor=color, face_alpha=0.45)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label))

    if uniform_p_opt_vline_ns:
        for n in uniform_p_opt_vline_ns:
            pu = _uniform_superposition_p_opt_htsp(int(n))
            if np.isfinite(pu) and pu > 0.0:
                all_y.append(float(pu))

    ax.set_xticks([float(n) for n in n_tick_vals])
    ax.set_xticklabels([str(n) for n in n_tick_vals])
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    _decorate_y_axis_from_values(
        ax,
        all_y,
        y_scale=y_scale,
        symlog_linthresh=symlog_linthresh,
        asinh_linear_width=asinh_linear_width,
        log_y_clip_upper=log_y_clip_upper,
        manual_y_limits=manual_y_limits,
    )
    if y_floor is not None:
        y_lo = float(y_floor)
        if y_scale == "log" and y_lo <= 0.0:
            y_lo = float(_P_OPT_LOG_AXIS_FLOOR)
        _, cur_hi = ax.get_ylim()
        ax.set_ylim(y_lo, cur_hi)
    if uniform_p_opt_vline_ns:
        ns_sorted = sorted({int(x) for x in uniform_p_opt_vline_ns})
        xs_u: list[float] = []
        ys_u: list[float] = []
        for n in ns_sorted:
            pu = _uniform_superposition_p_opt_htsp(n)
            if np.isfinite(pu) and pu > 0.0:
                xs_u.append(float(n))
                ys_u.append(float(pu))
        if xs_u:
            ax.plot(
                xs_u,
                ys_u,
                linestyle="--",
                color="dimgray",
                linewidth=1.45,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="dimgray",
                markeredgewidth=1.2,
                alpha=0.95,
                zorder=4,
                clip_on=False,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="dimgray",
                    linewidth=1.45,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor="dimgray",
                    markeredgewidth=1.2,
                    label=r"Uniform $P=1/(n-1)^{n-1}$",
                )
            )
            if y_scale == "log" and ys_u:
                y_lo_c, y_hi_c = ax.get_ylim()
                y_min_ref = float(min(ys_u))
                y_max_ref = float(max(ys_u))
                ax.set_ylim(
                    min(y_lo_c, y_min_ref / 1.35),
                    max(y_hi_c, y_max_ref * 1.08, 1.0),
                )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE_COMPACT if len(legend_handles) > 5 else LEGEND_FONTSIZE,
        )
    fig.tight_layout()
    return fig


def _plot_paired_four_series_boxplots_vs_p(
    *,
    x_labels: list[str],
    series: list[tuple[str, list[list[float]]]],
    y_label: str,
    x_axis_label: str,
    figsize: tuple[float, float] = (9.5, 5.2),
    strip_jitter: bool = True,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    """Up to four dodged box/strip series per *p* (e.g. V-QAOA/N-QAOA × two *n*)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    prop = plt.rcParams["axes.prop_cycle"].by_key()
    colors = prop["color"]
    n_g = len(x_labels)
    n_s = len(series)
    if_n = max(n_s - 1, 1)
    if n_s <= 3:
        dodge = 0.09
    else:
        dodge = 0.43 / float(if_n)
    half = 0.5 * float(n_s - 1) if n_s > 1 else 0.0
    box_w = dodge * 0.88
    jitter_w = min(box_w * 0.82, 0.055)
    all_y: list[float] = []
    legend_handles: list[Any] = []

    for rank, (leg_label, values_rows) in enumerate(series):
        color = colors[rank % len(colors)]
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.45, label=leg_label))
        for i in range(n_g):
            vals = values_rows[i] if i < len(values_rows) else []
            pos = float(i) + (float(rank) - half) * dodge
            for v in vals:
                if np.isfinite(v):
                    all_y.append(float(v))
            if not vals:
                continue
            if strip_jitter:
                _scatter_rho_instances_jittered(
                    ax,
                    pos,
                    vals,
                    color=color,
                    jitter_span=jitter_w,
                    rng=np.random.default_rng(8800 + rank * 41 + i * 19 + len(vals)),
                )
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_w,
                patch_artist=True,
                showfliers=not strip_jitter,
                boxprops=dict(linewidth=1.0, edgecolor=color),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=color,
                    markersize=3,
                    alpha=0.45,
                    linestyle="none",
                ),
            )
            _style_boxplot_patches_and_zorder(bp, facecolor=color, face_alpha=0.45)

    ax.set_xticks([float(i) for i in range(n_g)])
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    _decorate_y_axis_from_values(ax, all_y, y_scale=y_scale, symlog_linthresh=symlog_linthresh)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(handles=legend_handles, fontsize=LEGEND_FONTSIZE_COMPACT)
    fig.tight_layout()
    return fig


def _plot_grouped_bars_mean_std(
    *,
    x_labels: list[str],
    means_left: list[float],
    stds_left: list[float],
    means_right: list[float],
    stds_right: list[float],
    label_left: str,
    label_right: str,
    y_label: str,
    x_axis_label: str,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-5,
) -> Any:
    """Grouped bars with error bars (same layout as opt-steps plot).

    ``y_scale``: ``"linear"``, ``"log"`` (bars clipped upward to a small floor), or ``"symlog"``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(x_labels), dtype=np.float64)
    w = 0.36
    ml = np.array([float(v) if v == v else 0.0 for v in means_left], dtype=np.float64)
    mr = np.array([float(v) if v == v else 0.0 for v in means_right], dtype=np.float64)
    el = np.asarray(stds_left, dtype=np.float64)
    er = np.asarray(stds_right, dtype=np.float64)
    if y_scale == "log":
        ml = np.maximum(ml, _LOG_Y_FLOOR)
        mr = np.maximum(mr, _LOG_Y_FLOOR)
        el = np.minimum(el, np.maximum(ml - _LOG_Y_FLOOR, 0.0))
        er = np.minimum(er, np.maximum(mr - _LOG_Y_FLOOR, 0.0))
    ax.bar(x - w / 2, ml, w, yerr=el, capsize=3, label=label_left, color="#1f77b4")
    ax.bar(x + w / 2, mr, w, yerr=er, capsize=3, label=label_right, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    if y_scale == "log":
        ax.set_yscale("log")
    elif y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh, base=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    return fig
