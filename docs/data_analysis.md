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
| `{root}/images/*/` | Generated figures (`energy_history/`, `dashboards/`, `approx_ratio/`, `steps/`, `improvement/`, `p_opt/`) |

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
| `inst_n_precedences`, `inst_precedence_density`, `inst_prices_hotels_mean`, `inst_prices_hotels_std`, `inst_prices_hotels_range`, `inst_prices_travels_pos_mean`, `inst_prices_travels_pos_std` | Derived from `instance` (`data_analysis.instance_features`) |
| `oa_gamma`, `oa_beta` (lists in Parquet), `oa_gamma_json`, `oa_beta_json` | `metadata.optimal_angles` when `gamma`/`beta` length matches path or config `qaoa_depth` |
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

`paired_metrics` also carries **`runtime_per_energy_step`** (`runtime_seconds` / `n_energy_steps` when steps &gt; 0) for efficiency plots.

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
| `median_runtime_seconds` | Median of `runtime_seconds` (robust to outliers) |
| `mean_configs_evaluated` | Mean of `configs_evaluated` (NaNs ignored) |
| `mean_runtime_per_energy_step` | Mean of `runtime_seconds / n_energy_steps` where steps &gt; 0 |

Output: `processed/summary_by_config.csv`.

### 2.4 Aggregated energy curves (`energy_curves_agg`)

`aggregate_energy_curves`:

1. Filter `parse_ok ∧ solve_ok` with `n_energy_steps > 0`.
2. Group by `(n_cities, solver, formulation, qaoa_depth)`.
3. For each group and each row, open `path`, read `energy_history`, and **divide every value by `|ref_objective_value|`** for that row (BF optimum **objective** for that instance and formulation). Rows without a finite non-zero `ref_objective_value` are skipped.
4. Align **normalized** curves to the **longest** ``energy_history`` in each group (no default step cap).
5. For each step index, over non-NaN values: mean, sample `std` (`ddof=1`; 0 if only one curve), and `n_curves`; optionally **p25/p50/p75** unless you pass `--no-energy-curve-percentiles` to `data_analysis.metrics` or `data_analysis.pipeline` (plots only need `mean`/`std`).

Output (if data exists): `processed/energy_curves_agg.parquet` and `.csv`.

Columns include `step`, `mean`, `std`, `n_curves` plus grouping keys, and **`p25`/`p50`/`p75`** when percentiles are enabled (default).

### 2.5 Sample quality (`--sample-quality`)

`enrich_sample_quality` (optional):

For each row with `parse_ok`, `solve_ok`, and `has_final_samples`, opens JSON and reads `metadata.final_samples` (count → bitstring) and `instance`.

`_histogram_feasible_fraction`:

- **QUBO**: keys as `0/1` strings of length `n_available * n_available`, convert with `qubo_binary_to_sequence`, validate with `validate_solution_constraints_tqudo`.
- **TQUDO**: if key is a digit string of length `n_available`, interpret as sequence and validate the same way.
- Returns the fraction of total sample mass that decodes to a feasible tour.

Adds **`final_sample_feasible_mass`** (`None` if not applicable or parsing fails).

### 2.6 QAOA angle similarity (`angle_cohort_stats`, paired backends)

After `paired_metrics` is written, `data_analysis.metrics` builds:

- **`processed/angle_cohort_stats.parquet` / `.csv`**: per `(n_cities, solver, formulation, qaoa_depth)` cohort (excluding `brute_force`), summary of whether **normalized** \((\gamma,\beta)\) vectors are similar across instances: `mean_pairwise_cosine`, `std_pairwise_cosine`, per-layer `std_gamma_*` / `std_beta_*` (and JSON columns `std_gamma_json` / `std_beta_json`). If a cohort mixes different vector lengths (e.g. missing depth in path), only the **most frequent** length is kept (`angle_vector_dim_used`, `n_runs_angles_dim_consistent` vs `n_runs_with_angles`). If fewer than two runs remain, pairwise cosine statistics are NaN.
- **`processed/paired_angle_cudaq_tvirt_cirq_tqudo.parquet` / `.csv`**: inner join on `(n_cities, instance_key, qaoa_depth)` between CUDA-Q `tqudo_virtual` and Cirq `tqudo`; columns `cosine_pair`, `l2_delta` compare the concatenated unit vectors.
- **`processed/paired_angle_cudaq_qubo_tvirt.parquet` / `.csv`**: same join for CUDA-Q `qubo` vs CUDA-Q `tqudo_virtual` when both exist.

Empty tables are not written.

The pipeline **does not** analyze *simulated annealing*: rows with `solver == "simulated_annealing"` are dropped from `summary_by_config` and `energy_curves_agg` (the manifest / `paired_metrics` may still list those runs if present on disk).

## Phase 3: Plots (figures)

`run_plots` reads tables under `processed/plots_data/` (benchmark + energy history) and **also** reads aggregate tables directly from `processed/` for extended figures. It writes PNGs under `images/` **subfolders** (`energy_history/`, `dashboards/`, `approx_ratio/`, `steps/`, `improvement/`, `p_opt/`, `extended/`). A first run removes legacy PNGs that previously sat directly under `images/`.

Most figures have **no figure title**. Cohort and axis semantics are documented below.

**Notation in plots:** \(p\) = QAOA depth (repeated cost–mixer layers). \(f\) = **normalized** scalar objective stored in JSON (same scale as `energy_history`). \(\rho\) = `approx_ratio_real` = best feasible real cost found divided by **TQUDO** brute-force optimal real cost (see Section 2.2).

Requires `processed/plots_data` from `prepare_plots` (if missing, the command fails and asks to run `prepare_plots` first). Extended PNGs are skipped silently when the corresponding Parquet/CSV is absent.

### 0. Extended: instance features, efficiency, QAOA angles (`extended_plots.py`)

Source: `paired_metrics.parquet`, `summary_by_config.csv`, `angle_cohort_stats.parquet`, `paired_angle_*.parquet` in `processed/`. Only **feasible** QAOA rows enter the instance/efficiency scatters (brute force and simulated annealing excluded).

| PNG | What it shows |
|-----|----------------|
| `images/extended/instance_precedence_density_vs_rho.png` | Precedence density vs \(\rho\) (colour = backend/formulation). |
| `images/extended/efficiency_runtime_vs_rho.png` | Wall time vs \(\rho\) (log time). |
| `images/extended/efficiency_configs_evaluated_vs_rho.png` | `configs_evaluated` vs \(\rho\) when that field is populated; **otherwise** falls back to `n_energy_steps` (length of `energy_history`) with an updated axis title and footnote (QAOA JSON usually lacks `configs_evaluated`). |
| `images/extended/efficiency_mean_runtime_by_depth.png` | Mean runtime vs \(p\) from `summary_by_config` (one line per \(n\) + cohort). |
| `images/extended/angles_mean_pairwise_cosine_by_depth.png` | Cohort mean pairwise cosine vs \(p\) (higher ⇒ angles more similar across instances). |
| `images/extended/angles_cudaq_virt_cirq_cosine_hist.png` | Histogram of paired-instance cosine (CUDA-Q virt. vs Cirq). |
| `images/extended/angles_cudaq_virt_cirq_l2_hist.png` | Histogram of \(\ell_2\) gap between normalized angle vectors. |
| `images/extended/angles_cudaq_qubo_virt_cosine_hist.png` | Same cosine histogram for QUBO vs TQUDO virt. on CUDA-Q (if joined rows exist). |

**Better handled outside a single figure:** exact per-layer angle dispersion (`std_gamma_json` / `std_beta_json` in `angle_cohort_stats.csv`), partial correlations of several `inst_*` columns with \(\rho\), or calibration of \(P(\mathrm{opt})\) vs angle similarity — export `paired_metrics.parquet` / angle CSVs into a notebook and compute correlations or regressions there.

### 1. Energy history (`energy_plots.py`, multiple PNGs; no SA)

Source: `energy_curves_agg.parquet` (`mean` and `std` per step, per `(n_cities, solver, formulation, qaoa_depth)`). **One file per QAOA depth** \(p\) (filename suffix `_p{p}`). Values in the table are **already** per-instance \(E / |E^*_\mathrm{inst}|\); the line is the **mean across instances** at that \(p\); the band is **mean ± σ** (sample std across normalized curves, `ddof=1`).

**Horizontal dashes:** brute-force optimum at **\(\pm 1\)** after the same per-instance scaling (median sign of `ref_objective_value` in the cohort; usually \(-1\) if all optima are negative).

#### `images/energy_history/cudaq_qubo_tvirt_n5_p{p}.png`

- **Cohort:** `n_cities = 5`; CUDA-Q `qubo` and `tqudo_virtual`; fixed \(p\). Legend: “QUBO” and **“V-QAOA”**.
- **X axis:** Step. **Y axis:** shared; each series is per-instance \(f/|f^*|\) for its formulation (comparable after normalization).

#### `images/energy_history/cirq_tqudo_vs_cq_tvirt_n5_n9_p{p}.png`

- **Cohort:** Cirq `tqudo` (**N-QAOA**) and CUDA-Q `tqudo_virtual` (**V-QAOA**) at **`n \in \{5,9\}`** (up to four series); fixed \(p\). Missing \((n,\text{solver})\) cells are skipped.
- **Y axis:** \(f/|f^*|\); each series uses its cohort’s BF objective as \(f^*\).

#### `images/energy_history/cirq_tqudo_by_n_p{p}.png`

- **Cohort:** `solver == cirq`, `formulation == tqudo`; one series per `n_cities` at fixed \(p\).
- **Y axis:** \(f/|f^*|\) with \(f^*\) from the BF reference for that \(n\).

### 2. Comparison dashboards and approximation ratio (`data_analysis.benchmark.run`)

Requires `paired_metrics.parquet`. **2×2 dashboards use paired rows**: same `instance_key` and \(p\); real-cost optimality vs brute-force TQUDO (`ref_real_cost`).

#### 2×2 dashboard — `images/dashboards/cudaq_qubo_vs_tvirt_n5.png`

- **Pair:** left QUBO, right **V-QAOA** (CUDA-Q `tqudo_virtual`); `n_cities = 5`; grouped bars for \(p \in \{1,2,3\}\).
- **Top-left:** **stacked** counts (optimal / feasible suboptimal / infeasible) per side; Y “Instances”.
- **Top-right:** among instances **feasible on both sides**, percentage where **real cost** is lower on left, right, or tie; Y “% (both feasible)”.
- **Bottom-left:** percent of **paired** total where only one side is feasible (short legend labels).
- **Bottom-right:** asymmetric **optimality** (optimal on one side only per `ref_real_cost`).
- **Shared X:** \(p\).

#### 2×2 dashboard — `images/dashboards/cudaq_tvirt_vs_cirq_n5.png`

Same layout; pair **V-QAOA** (left) vs **N-QAOA** (right), `n_cities = 5`.

#### 2×2 dashboard — `images/dashboards/cudaq_tvirt_vs_cirq_n9.png`

Same layout and pairing as above, but **`n_cities = 9`**. This needs an **inner join** on `(instance_key, qaoa_depth)` between CUDA-Q `tqudo_virtual` and Cirq `tqudo` solution rows at that size. If there are no Cirq runs at \(n=9\) (common in layouts that stop native qudits at \(n=8\)), paired rows are empty and the dashboard has no data.

#### Mean ratio vs \(p\) — `images/approx_ratio/n5_qubo_tvirt_cirq_vs_p.png`

- **Not** paired cohorts: three **independent** series (feasible rows with finite `approx_ratio_real` each), one deduped row per `(instance_key, p)` per formulation; BF TQUDO reference.
- **X:** \(p\); **Y:** \(\rho\) (mean ± σ); gray “ρ = 1” = reference optimum.
- **Series labels:** “QUBO”, “V-QAOA”, “N-QAOA” (all `n_cities = 5`).
- Error bars are sample std (`ddof=1`) within each \(p\).

#### Mean ratio vs \(n\) — `images/approx_ratio/rho_vs_n_by_p.png`

- **X:** \(n\) (cities). Each \(p\) is a series with slight **dodge** on X when multiple depths exist.
- **Project convention:** for \(n \le 8\) use **Cirq** `tqudo`; for **\(n=9\)**, **CUDA-Q** `tqudo_virtual` (no native Cirq at that size in the usual disk layout).
- **Y:** \(\rho\) (mean ± σ); reference \(\rho = 1\).

#### Steps to first `energy_history` minimum (per-solver optimality)

**Metric:** 1-based step index: first optimizer step where `energy_history` reaches its **global minimum** for that run (`first_optimizer_step_reaching_min_energy`). Values are read at plot time from each solution JSON (`path` or paired `path_left` / `path_right`).

**Cohort:** only runs that are **optimal in real cost** vs brute-force TQUDO (`is_optimal_vs_ref` on `real_cost`, `ref_real_cost`, `feasible`) **for that solver** enter the mean/σ on that side. Paired figures still use an **inner join** on `(n_cities, instance_key, qaoa_depth)`, but the two sides **need not** both be optimal; statistics are computed **independently** over qualifying rows on the left and right.

#### `images/steps/cudaq_tvirt_vs_qubo_n5_vs_p.png`

- **Pair:** CUDA-Q `tqudo_virtual` vs CUDA-Q `qubo`, `n_cities = 5` (merge order: left **“V-QAOA”**, right “QUBO”).
- **X:** \(p\); **Y:** mean ± σ (sample std, `ddof=1`) of step counts; **markers + error bars** with slight horizontal dodge between series at each \(p\).

#### `images/steps/cudaq_tvirt_vs_cirq_n5_n9_vs_p.png`

- **Two subplots:** \(n = 5\) and \(n = 9\); same pair as the V-QAOA vs N-QAOA dashboards (**V-QAOA** vs **N-QAOA**); **shared** vertical scale.
- **X:** \(p\); **Y:** same step definition and per-side optimality filter as above.

#### `images/steps/cirq_tqudo_firstmin_steps_vs_n_by_p.png`

- **Cohort:** Cirq `tqudo` only (no cross-backend pairing); deduped row per `(instance_key, p)` at each \(n\).
- **X:** \(n \in \{5,6,7,8,9\}\), with **dodge** between series for each \(p \in \{1,2,3\}\) (same layout idiom as mean \(\rho\) vs \(n\)).
- **Y:** mean ± σ of step counts among runs optimal vs `ref_real_cost` at each \((n, p)\).
- **Emitted only** when at least one \((n,p)\) cell has qualifying runs.

Older grouped-bar figures named `*_opt_steps_both_optimal_*` (both sides optimal in each pair) are **no longer produced**.

### 3. Ground-state sample probability and improvements (`data_analysis.benchmark.run`, `optimal_sample_mass.py`)

These figures re-open solution JSON at plot time (in addition to `paired_metrics`). They need **`raw/solutions/brute_force/tqudo/n_{n}/instance_{k}.json`** for the reference tour and **`solver_output.metadata.initial_samples` / `final_samples`** where applicable.

**Optimal histogram key:** the brute-force TQUDO metadata field **`best_feasible_sequence`**, or **`best_sequence`** if the former is absent. That integer list is turned into the same string key used in sample dicts: **Cirq native `tqudo`** — dash-separated qudits (e.g. `"0-2-1-3"`); **`tqudo_virtual`** (CUDA-Q) — contiguous `0`/`1` string via `utils.costs_batch.qudit_sequence_to_bitstring` (little-endian blocks, matching `bitstring_to_qudit_sequence`).

**\(P(\mathrm{opt})\)** for a histogram \(H\): `H[key] / sum(H.values())` (not the YAML shot count alone). If the key is missing, the mass is 0.

**Relative energy improvement** in paired metrics is already **`energy_improvement_rel`** (§2.2): \((E_0 - E^\star)/|E_0|\) with \(E_0 =\) `initial_energy` in JSON metadata.

**\(\Delta P(\mathrm{opt})\)** per run: \(P\) from `final_samples` minus \(P\) from `initial_samples` for the same BF key; rows without both histograms are skipped.

**Y-axis scaling:** \(P(\mathrm{opt})\) plots use **log *y*** (means clipped to a tiny positive floor when needed so error bars stay valid). \(\Delta P(\mathrm{opt})\) plots use **symlog *y*** so negative changes remain visible.

#### `images/p_opt/cirq_tqudo_popt_vs_n_by_p.png`

- **Cohort:** Cirq `tqudo`, `solve_ok`, `has_final_samples`; \(p \in \{1,2,3\}\).
- **X:** \(n\) (cities); series are **dodged** per \(p\).
- **Y:** mean ± σ of \(P(\mathrm{opt})\) over instances at each \((n, p)\); **log scale**.

#### `images/p_opt/n5_cirq_vs_cq_tvirt_popt_vs_p.png`

- **`n_cities = 5`**: two unpaired series — Cirq `tqudo` vs CUDA-Q `tqudo_virtual` — mean ± σ of \(P(\mathrm{opt})\) vs \(p\); **log scale**.

#### `images/improvement/cirq_tqudo_rel_energy_vs_n_by_p.png`

- **Cohort:** Cirq `tqudo`; **Y:** mean ± σ of **`energy_improvement_rel`** vs \(n\) (three series for \(p\), dodged X).

#### `images/p_opt/cirq_tqudo_delta_popt_vs_n_by_p.png`

- **Cohort:** Cirq `tqudo` with both **`has_initial_samples`** and **`has_final_samples`**; **Y:** mean ± σ of \(\Delta P(\mathrm{opt})\) vs \(n\) per \(p\); **symlog scale**.

#### `images/improvement/paired_n5_cq_cirq_rel_energy_vs_p.png`

- **Inner join** on `(n_cities, instance_key, qaoa_depth)` with **`n = 5`**: **V-QAOA** (left) vs **N-QAOA** (right), same pairing as the 2×2 V-QAOA vs N-QAOA dashboard.
- **X:** \(p\); **Y:** mean ± σ of **`energy_improvement_rel`** on each side (grouped bars).

#### `images/p_opt/paired_n5_cq_cirq_delta_popt_vs_p.png`

- Same paired cohort as above; **Y:** mean ± σ of \(\Delta P(\mathrm{opt})\) from each side’s JSON (initial vs final samples); **symlog scale**.

## Programmatic API

From code:

```python
from data_analysis.pipeline import run_pipeline, process_raw_results
from pathlib import Path

run_pipeline(Path("output"), manifest_format="parquet", sample_quality=False, skip_plots=False)
# Compatibility: raw_dir=.../output/raw, processed_dir must be .../output/processed (same output root)
process_raw_results(Path("output/raw"), Path("output/processed"))
```

`data_analysis.__init__` lazily re-exports `process_raw_results` and `run_pipeline`.

## Limitations and good practice

- Without **brute_force** rows for `(n_cities, instance_key, formulation)`, `approx_ratio_*` ratios are undefined (NaN).
- Ground-state **\(P(\mathrm{opt})\)** and **\(\Delta P(\mathrm{opt})\)** plots need brute-force **TQUDO** JSON for each instance and, for ΔP, both **initial** and **final** sample dicts in the solution file; missing files or keys drop that sample from the aggregate.
- Energy curves include all aligned step indices up to the longest history in each `(n_cities, solver, formulation, qaoa_depth)` group; each row only contributes where it has data (shorter runs leave NaNs that are dropped per-step for the aggregate).
