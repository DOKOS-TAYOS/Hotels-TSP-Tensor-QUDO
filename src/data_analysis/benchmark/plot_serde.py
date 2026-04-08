"""Serialize / deserialize per-figure tables for ``processed/plots_data``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def plot_kwargs_jsonable(plot_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Recursively turn tuples into lists for JSON ``plot_kwargs``."""

    def _conv(x: Any) -> Any:
        if isinstance(x, tuple):
            return [_conv(i) for i in x]
        if isinstance(x, list):
            return [_conv(i) for i in x]
        if isinstance(x, dict):
            return {str(k): _conv(v) for k, v in x.items()}
        return x

    return {str(k): _conv(v) for k, v in plot_kwargs.items()}


def coerce_plot_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Restore types after JSON round-trip for matplotlib call sites."""
    out = dict(raw)
    fs = out.get("figsize")
    if isinstance(fs, list) and len(fs) == 2:
        out["figsize"] = (float(fs[0]), float(fs[1]))
    for k in ("uniform_p_opt_hline_ns", "uniform_qubo_p_opt_hline_ns"):
        v = out.get(k)
        if isinstance(v, list):
            out[k] = tuple(int(x) for x in v)
    return out


def meta_path_for_parquet(parquet_path: Path) -> Path:
    return parquet_path.parent / f"{parquet_path.stem}.meta.json"


def write_meta(parquet_path: Path, meta: dict[str, Any]) -> None:
    meta_path_for_parquet(parquet_path).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def read_meta(parquet_path: Path) -> dict[str, Any]:
    p = meta_path_for_parquet(parquet_path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_dashboard_stats(
    parquet_path: Path,
    stats_list: list[dict[str, float | int]],
    *,
    x_labels: list[str],
    label_left: str,
    label_right: str,
    x_axis_label: str,
    other_panels_stats_stop: int | None = None,
) -> None:
    import pandas as pd

    df = pd.DataFrame(stats_list)
    df.insert(0, "x_label", x_labels)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    meta: dict[str, Any] = {
        "kind": "dashboard",
        "label_left": label_left,
        "label_right": label_right,
        "x_axis_label": x_axis_label,
    }
    if other_panels_stats_stop is not None:
        meta["other_panels_stats_stop"] = int(other_panels_stats_stop)
    write_meta(parquet_path, meta)


def read_dashboard_stats(
    parquet_path: Path,
) -> tuple[list[str], list[dict[str, float | int]], dict[str, Any]]:
    import pandas as pd

    meta = read_meta(parquet_path)
    df = pd.read_parquet(parquet_path)
    x_labels = [str(x) for x in df["x_label"].tolist()]
    stat_cols = [c for c in df.columns if c != "x_label"]
    stats_list: list[dict[str, float | int]] = []
    for _, row in df.iterrows():
        stats_list.append({c: row[c] for c in stat_cols})
    return x_labels, stats_list, meta


def write_box_vs_p_long(
    parquet_path: Path,
    series: list[tuple[str, dict[int, list[float]]]],
    *,
    plot_kwargs: dict[str, Any],
) -> None:
    import pandas as pd

    rows: list[dict[str, object]] = []
    series_order = [lab for lab, _ in series]
    for label, depth_vals in series:
        for p in sorted(depth_vals.keys()):
            for v in depth_vals[int(p)]:
                rows.append({"series_label": label, "p": int(p), "value": float(v)})
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    write_meta(
        parquet_path,
        {
            "kind": "box_vs_p",
            "series_order": series_order,
            "plot_kwargs": plot_kwargs_jsonable(plot_kwargs),
        },
    )


def read_box_vs_p_long(
    parquet_path: Path,
) -> tuple[list[tuple[str, dict[int, list[float]]]], dict[str, Any]]:
    import pandas as pd

    meta = read_meta(parquet_path)
    df = pd.read_parquet(parquet_path)
    order: list[str] = list(meta["series_order"])
    out_map: dict[str, dict[int, list[float]]] = {lab: {} for lab in order}
    for lab in order:
        sub = df[df["series_label"] == lab]
        for p in sorted(sub["p"].unique()):
            pv = int(p)
            vals = sub.loc[sub["p"] == p, "value"].astype(float).tolist()
            out_map[lab][pv] = [float(v) for v in vals]
    series = [(lab, out_map[lab]) for lab in order]
    return series, meta


def write_triplet_series_long(
    parquet_path: Path,
    series: list[tuple[str, list[float], list[list[float]]]],
    *,
    plot_kwargs: dict[str, Any],
    kind: str = "triplet_vs_x",
    triplet_plot: str = "dodged_ncities",
) -> None:
    import pandas as pd

    rows: list[dict[str, object]] = []
    series_order = [lab for lab, _, _ in series]
    for label, xs, datas in series:
        for x, vals in zip(xs, datas, strict=True):
            for v in vals:
                rows.append({"series_label": label, "x_pos": float(x), "value": float(v)})
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    write_meta(
        parquet_path,
        {
            "kind": kind,
            "triplet_plot": triplet_plot,
            "series_order": series_order,
            "plot_kwargs": plot_kwargs_jsonable(plot_kwargs),
        },
    )


def read_triplet_series_long(
    parquet_path: Path,
) -> tuple[list[tuple[str, list[float], list[list[float]]]], dict[str, Any]]:
    import pandas as pd

    meta = read_meta(parquet_path)
    df = pd.read_parquet(parquet_path)
    order: list[str] = list(meta["series_order"])
    out: list[tuple[str, list[float], list[list[float]]]] = []
    for lab in order:
        sub = df[df["series_label"] == lab]
        xs_sorted = sorted(sub["x_pos"].unique())
        xs: list[float] = []
        datas: list[list[float]] = []
        for x in xs_sorted:
            vals = sub.loc[sub["x_pos"] == x, "value"].astype(float).tolist()
            xs.append(float(x))
            datas.append([float(v) for v in vals])
        out.append((lab, xs, datas))
    return out, meta


def write_paired_four_vs_p(
    parquet_path: Path,
    *,
    x_labels: list[str],
    series: list[tuple[str, list[list[float]]]],
    plot_kwargs: dict[str, Any],
) -> None:
    import pandas as pd

    rows: list[dict[str, object]] = []
    series_order = [lab for lab, _ in series]
    n_g = len(x_labels)
    for label, value_rows in series:
        for i in range(n_g):
            vals = value_rows[i] if i < len(value_rows) else []
            for v in vals:
                rows.append({"series_label": label, "p_index": int(i), "value": float(v)})
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    write_meta(
        parquet_path,
        {
            "kind": "paired_four_vs_p",
            "x_labels": x_labels,
            "series_order": series_order,
            "plot_kwargs": plot_kwargs_jsonable(plot_kwargs),
        },
    )


def read_paired_four_vs_p(
    parquet_path: Path,
) -> tuple[list[str], list[tuple[str, list[list[float]]]], dict[str, Any]]:
    import pandas as pd

    meta = read_meta(parquet_path)
    x_labels = [str(x) for x in meta["x_labels"]]
    order: list[str] = list(meta["series_order"])
    df = pd.read_parquet(parquet_path)
    n_g = len(x_labels)
    series: list[tuple[str, list[list[float]]]] = []
    for lab in order:
        sub = df[df["series_label"] == lab]
        rows_out: list[list[float]] = []
        for i in range(n_g):
            vals = sub.loc[sub["p_index"] == i, "value"].astype(float).tolist()
            rows_out.append([float(v) for v in vals])
        series.append((lab, rows_out))
    return x_labels, series, meta
