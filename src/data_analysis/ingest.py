"""CLI: scan ``output/raw`` and write ``output/processed/manifest``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_analysis.records import json_row, manifest_empty_schema_row
from data_analysis.scan import iter_raw_json_files
from data_analysis.output_paths import build_output_layout


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ingest requires pandas (install project extra: pip install -e '.[analysis]')."
        ) from exc


def run_ingest(output_root: Path, fmt: str) -> Path:
    """Build manifest from all JSON under ``raw/`` and write to ``processed/``."""
    _require_pandas()
    import pandas as pd

    layout = build_output_layout(output_root)
    layout.processed.mkdir(parents=True, exist_ok=True)
    raw_dir = layout.raw
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    rows: list[dict[str, object]] = []
    for path in sorted(iter_raw_json_files(raw_dir)):
        rows.append(json_row(path, output_root))

    if not rows:
        df = pd.DataFrame([manifest_empty_schema_row()]).iloc[0:0]
    else:
        df = pd.DataFrame(rows)
    if fmt == "parquet":
        out = layout.processed / "manifest.parquet"
        df.to_parquet(out, index=False)
    else:
        out = layout.processed / "manifest.csv"
        df.to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build manifest from raw experiment JSON.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Output root (contains raw/ and processed/).",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Manifest format (default: parquet).",
    )
    args = parser.parse_args(argv)
    out = run_ingest(args.output_root.resolve(), args.format)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
