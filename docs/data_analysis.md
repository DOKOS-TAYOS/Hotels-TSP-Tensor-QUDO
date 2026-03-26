# Data analysis (`data_analysis`)

This document describes what the `data_analysis` package does: how it walks experiment JSON files, which columns it extracts, which metrics it derives, which aggregates it builds, and which figures it generates.

## Dependencies and setup

The code lives under `src/data_analysis/`. Use the project **analysis** extra (installs `pandas`; figures also need `matplotlib`).

```bash
pip install -e '.[analysis]'
```

Main CLI entry points:

| Module | Command | Role |
|--------|---------|------|
| `data_analysis.ingest` | `python -m data_analysis.ingest --output-root output` | Build the manifest |
| `data_analysis.metrics` | `python -m data_analysis.metrics --output-root output` | Derived tables and stats JSON |
| `data_analysis.plot` | `python -m data_analysis.plot --output-root output` | PNG figures |
| `data_analysis.pipeline` | `python -m data_analysis.pipeline --output-root output` | Full sequence (ingest → metrics → plots) |

Useful options:

- `--format parquet|csv` on ingest (default `parquet`).
- `--sample-quality` on metrics (slower: re-reads JSON for feasibility histogram over final samples).
- `--skip-plots` on pipeline for tables only.

## Expected directory layout

The module assumes an **output root** (default `output/`) with this layout (`utils.output_paths.build_output_layout`):

| Path | Contents |
|------|----------|
| `{root}/raw/solutions/...` | Solution JSON produced by the experiment workflow |
| `{root}/processed/` | Manifest and aggregated tables |
| `{root}/images/` | Generated figures |

File discovery **only** considers JSON under `raw/solutions/**/*.json` (`data_analysis.scan.iter_raw_json_files`).

## Phase 1: Ingest (manifest)

### File discovery

Every `.json` under `raw/solutions/` is sorted by path and becomes one manifest row.

### On-disk path convention

For rows with `layout == "disk"`, the path relative to the output root must match:

`raw/solutions/{solver}/{formulation}/n_{n_cities}/[qaoa_depth/]instance_{k}.json`

- `qaoa_depth` is an **optional** numeric path segment (e.g. simulated annealing may omit it).
- `instance_key` is taken from the number in `instance_{k}.json`.

If the path does not match, the row may stay `layout: unknown` without path-inferred fields.

### Fields per row (`json_row`)

Each JSON is opened and validated:

- **`parse_ok`**: valid JSON with an object at the root.
- **`solve_ok`**: `solver_output` exists as an object and has **no** `error` key. If `solver_output` is missing or there is an error, `parse_ok` may still be true while `solve_ok` is false (solver failure stored as a valid file).

Typical fields (non-exhaustive; see `manifest_empty_schema_row` in `records.py` for the empty schema):

| Column | Source |
|--------|--------|
| `path` | Relative path to output root |
| `layout` | `disk` or `unknown` |
| `solver`, `formulation`, `n_cities`, `instance_key`, `qaoa_depth` | Path and/or `solver_config` |
| `n_cities_json` | `instance.n_cities` in the JSON |
| `solver_config_solver`, `solver_config_formulation`, `seed` | `solver_config` |
| `noise_enabled` | `solver_config.noise.enabled` if dict |
| `instance_index` | Optional top-level field |
| `feasible`, `objective_value`, `runtime_seconds` | `solver_output` |
| `real_cost`, `n_energy_steps`, `has_final_samples`, `has_initial_samples`, `initial_energy`, `best_feasible_objective_value`, `best_feasible_real_cost`, `configs_evaluated` | `solver_output.metadata` |
| `solver_error`, `error_message` | Parse or solver errors |

Fill rules:

- `n_energy_steps` = length of `metadata.energy_history` if list, else 0.
- If path `n_cities` is missing but `n_cities_json` exists, copy it.
- If `instance_key` is missing but `instance_index` exists, use `instance_index + 1`.

### Ingest output

- `processed/manifest.parquet` (or `manifest.csv` if requested).

If there are no JSON files, an empty DataFrame with the same column schema is written.

## Phase 2: Metrics (paired metrics and aggregates)

The manifest is read (`manifest.parquet` preferred, else `manifest.csv`). Columns `parse_ok` and `solve_ok` are normalized to booleans; if `solve_ok` is missing in old manifests, it is inferred as `parse_ok` ∧ no `solver_error`.

### 2.1 Brute-force reference

`_reference_bruteforce`:

- Keeps rows with `parse_ok`, `solve_ok`, and `solver == "brute_force"`.
- **Real-cost optimum (`ref_real_cost`)**: only `formulation == "tqudo"` (TQUDO enumeration covers the `n_cities` of interest). For each `(n_cities, instance_key)` it takes the **last** row after sorting by `path`.
- **`ref_real_cost`**: `best_feasible_real_cost` if not null; otherwise `real_cost`.
- Objective references for ratios / curves: if `brute_force` + `qubo` rows exist for the same instance, their `objective_value` is kept as the QUBO objective reference.

Without brute-force TQUDO, reference columns in pairing stay NaN.

### 2.2 Paired table (`paired_metrics`)

`build_paired_metrics` merges the manifest with the reference on `(n_cities, instance_key)` and defines:

- **`ref_objective_value`**: brute-force **QUBO** optimum on `formulation == "qubo"` rows (if QUBO JSON exists); brute-force **TQUDO** optimum on `formulation` ∈ {`tqudo`, `tqudo_virtual`}. Units match each run’s formulation.

| Column | Definition |
|--------|------------|
| `approx_ratio_real` | `real_cost / ref_real_cost` if both exist and `ref_real_cost > 0`; else NaN |
| `approx_ratio_objective` | `objective_value / ref_objective_value` if denominator ≠ 0 |
| `energy_improvement_rel` | `(initial_energy - objective_value) / |initial_energy|` if `initial_energy` ≠ 0 and both defined |

Interpretation:

- **`approx_ratio_real` ≈ 1** means real cost near the reference optimum (same optimal tour as brute-force TQUDO on that instance).
- **`energy_improvement_rel`** is relative reduction of the scalar objective from the optimizer start (same units as stored in JSON, usually scaled).

Outputs:

- `processed/paired_metrics.parquet` and `paired_metrics.csv`.

### 2.3 Per-configuration summary (`summary_by_config`)

`build_summary_by_config` keeps only `parse_ok ∧ solve_ok` and groups by:

`(n_cities, solver, formulation, qaoa_depth)`

(Missing columns filled with NaN so grouping works.)

Per-group aggregates:

| Metric | Meaning |
|--------|---------|
| `n_runs` | Row count |
| `feas_rate` | Mean of `feasible` with NaN as false |
| `mean_runtime` | Mean of `runtime_seconds` |
| `mean_objective` | Mean of `objective_value` |
| `mean_real_cost` | Mean of `real_cost` |
| `mean_approx_ratio_real` | Mean of `approx_ratio_real` |
| `mean_energy_steps` | Mean of `n_energy_steps` |

Output: `processed/summary_by_config.csv`.

### 2.4 Aggregated energy curves (`energy_curves_agg`)

`aggregate_energy_curves`:

1. Filter `parse_ok ∧ solve_ok` with `n_energy_steps > 0`.
2. Group by `(n_cities, solver, formulation, qaoa_depth)`.
3. For each group, open each relative `path` JSON and read `solver_output.metadata.energy_history` as a float list.
4. Align curves to `min(500, max_length)` (default `max_len_cap` 500).
5. For each step index, over non-NaN values: p25/p50/p75, mean, sample `std` (`ddof=1`; 0 if only one curve), and `n_curves`.

Output (if data exists): `processed/energy_curves_agg.parquet` and `.csv`.

Columns include `step`, `p25`, `p50`, `p75`, `mean`, `std`, `n_curves` plus grouping keys.

### 2.5 Sample quality (`--sample-quality`)

`enrich_sample_quality` (optional):

For each row with `parse_ok`, `solve_ok`, and `has_final_samples`, opens JSON and reads `metadata.final_samples` (count → bitstring) and `instance`.

`_histogram_feasible_fraction`:

- **QUBO**: keys as `0/1` strings of length `n_available * n_available`, convert with `qubo_binary_to_sequence`, validate with `validate_solution_constraints_tqudo`.
- **TQUDO**: if key is a digit string of length `n_available`, interpret as sequence and validate the same way.
- Returns the fraction of total sample mass that decodes to a feasible tour.

Adds **`final_sample_feasible_mass`** (`None` if not applicable or parsing fails).

The pipeline **does not** analyze *simulated annealing*: rows with `solver == "simulated_annealing"` are dropped from `summary_by_config` and `energy_curves_agg` (the manifest / `paired_metrics` may still list those runs if present on disk).

## Phase 3: Plots (figures)

`run_plots` reads tables already in `processed/` and writes PNGs under `images/`.

Figures have **no title** (`title`/`suptitle`): cohort, metric definitions, and how to read axes and legends are documented here.

**Notation in plots:** \(p\) = QAOA depth (repeated cost–mixer layers). \(f\) = **normalized** scalar objective stored in JSON (same scale as `energy_history`). \(\rho\) = `approx_ratio_real` = best feasible real cost found divided by **TQUDO** brute-force optimal real cost (see Section 2.2).

Requires `paired_metrics.parquet` and/or `energy_curves_agg.parquet` (if neither exists, the command fails and asks to run metrics first).

### 1. Energy history (`energy_plots.py`, three PNGs; no SA)

Source: `energy_curves_agg.parquet` (`mean` and `std` per step and depth). For each X step, the **line** is the **weighted** mean over depths \(p\) with weights `n_curves`. The **band** is **mean ± σ**; σ at each depth is the sample std across curves (`ddof=1`); combining depths uses the same weights as the mean.

**Horizontal dashed lines**: mean of `ref_objective_value` over distinct instances (`instance_key`) — brute-force optimum **objective** in the **same formulation** as the series — **not** tour real cost.

#### `energy_history_mean_cudaq_qubo_vs_tqudo_virtual_n5.png`

- **Cohort:** `n_cities = 5`; CUDA-Q `qubo` and `tqudo_virtual`.
- **X axis:** optimizer step (short label: “Step”).
- **Twin Y axes:** left = QUBO mean ± σ trace; right = TQUDO virtual. Scales **differ** (per-formulation normalization); do not compare absolute values across axes — compare curve shape and distance to the reference dash **on that axis**.
- **Legend:** “QUBO” and “TQUDO virt.” curves; dashes usually omit legend (match series color).
- **Use:** compare optimizer convergence across CUDA-Q formulations when native objective scales are not numerically comparable.

#### `energy_history_mean_cirq_tqudo_vs_cudaq_tvirt_n5.png`

- **Cohort:** `n_cities = 5`; Cirq `tqudo` and CUDA-Q `tqudo_virtual`.
- **X axis:** Step.
- **Y axis:** \(f\) (mean ± σ); **single** scale (both are TQUDO).
- **Legend:** “Cirq TQUDO”, “CQ virt.”, “BF optimum” for the dashed line (mean BF TQUDO objective for that \(n\)).
- **Use:** compare backends under the same TQUDO-style encoding on one energy axis.

#### `energy_history_mean_cirq_tqudo_by_ncities.png`

- **Cohort:** all aggregated rows with `solver == cirq` and `formulation == tqudo`; one series per `n_cities` in `energy_curves_agg`.
- **X axis:** Step; **Y axis:** \(f\) (mean ± σ).
- **Legend:** “n = …” by color; same-color dashed line = mean BF TQUDO reference for that \(n\).
- **Use:** see how typical energy traces scale with problem size (native Cirq).

### 2. Comparison dashboards and approximation ratio (`benchmark_plots.py`)

Requires `paired_metrics.parquet`. **2×2 dashboards use paired rows**: same `instance_key` and \(p\); real-cost optimality vs brute-force TQUDO (`ref_real_cost`).

#### 2×2 dashboard — `comparison_cudaq_qubo_vs_tqudo_virtual_by_qaoa_depth.png`

- **Pair:** left CUDA-Q QUBO, right CUDA-Q TQUDO virtual; `n_cities = 5`; grouped bars for \(p \in \{1,2,3\}\).
- **Top-left:** **stacked** counts (optimal / feasible suboptimal / infeasible) per side; Y “Instances”.
- **Top-right:** among instances **feasible on both sides**, percentage where **real cost** is lower on left, right, or tie; Y “% (both feasible)”.
- **Bottom-left:** percent of **paired** total where only one side is feasible (short legend labels).
- **Bottom-right:** asymmetric **optimality** (optimal on one side only per `ref_real_cost`).
- **Shared X:** \(p\).

#### 2×2 dashboard — `comparison_cudaq_tqudo_virtual_vs_cirq_tqudo_by_qaoa_depth.png`

Same layout; pair **CQ virt.** (left) vs **Cirq TQUDO** (right), `n_cities = 5`.

#### Mean ratio vs \(p\) — `comparison_mean_approx_ratio_cudaq_qubo_cudaq_tvirt_cirq_tqudo_n5_by_qaoa_depth.png`

- **Not** paired cohorts: three **independent** series (feasible rows with finite `approx_ratio_real` each), one deduped row per `(instance_key, p)` per formulation; BF TQUDO reference.
- **X:** \(p\); **Y:** \(\rho\) (mean ± σ); gray “ρ = 1” = reference optimum.
- **Series labels:** “QUBO”, “TQUDO virt.”, “Cirq TQUDO” (all `n_cities = 5`).
- Error bars are sample std (`ddof=1`) within each \(p\).

#### Mean ratio vs \(n\) — `comparison_mean_approx_ratio_cirq_tqudo_n5_n8_cudaq_tvirt_n9_by_ncities.png`

- **X:** \(n\) (cities). Each \(p\) is a series with slight **dodge** on X when multiple depths exist.
- **Project convention:** for \(n \le 8\) use **Cirq** `tqudo`; for **\(n=9\)**, **CUDA-Q** `tqudo_virtual` (no native Cirq at that size in the usual disk layout).
- **Y:** \(\rho\) (mean ± σ); reference \(\rho = 1\).

#### Steps to trace minimum — `comparison_cudaq_qubo_vs_tqudo_virtual_opt_steps_both_optimal_by_qaoa_depth.png`

- Only pairs where **both** sides are **optimal** in real cost vs BF TQUDO.
- **X:** \(p\); **Y:** “Steps (mean ± σ)”: 1-based step count until `energy_history` **first** hits its global minimum in that JSON (read at plot time from `path_left` / `path_right`).
- **Legend:** “QUBO” vs “TQUDO virt.” (grouped bars).

#### Steps to trace minimum — `comparison_cudaq_tqudo_virtual_vs_cirq_tqudo_opt_steps_both_optimal_by_qaoa_depth.png`

Same metric and filters; pair **CQ virt.** vs **Cirq TQUDO**.

## Programmatic API

From code:

```python
from data_analysis.pipeline import run_pipeline, process_raw_results
from pathlib import Path

run_pipeline(Path("output"), manifest_format="parquet", sample_quality=False, skip_plots=False)
# Compatibility: raw_dir=.../output/raw, processed_dir=.../output/processed
process_raw_results(Path("output/raw"), Path("output/processed"))
```

`data_analysis.__init__` lazily re-exports `process_raw_results` and `run_pipeline`.

## Limitations and good practice

- Without **brute_force** rows for `(n_cities, instance_key, formulation)`, `approx_ratio_*` ratios are undefined (NaN).
- Energy curves cap at 500 steps by default and only include steps where at least one curve has data.
