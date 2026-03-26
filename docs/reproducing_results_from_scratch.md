# Reproducing results from scratch

Minimal command sequence from a clean clone to **instances**, **solutions**, **processed tables**, **figures**, and the **static dashboard**. Requires **Linux** (WSL2 OK); CUDA-Q needs an **NVIDIA GPU** and the `cudaq` extra.

---

## 1. Install

```bash
git clone <your-fork-or-upstream> Hotels-TSP-Tensor-QUDO
cd Hotels-TSP-Tensor-QUDO
./install.sh                    # default: dev,ui,cudaq
# Include analysis + figures:
./install.sh dev,ui,cudaq,analysis
```

For **CPU-only** (no GPU), use Cirq / SA / brute_force instead:

```bash
./install.sh dev,ui,cirq,analysis
```

Copy / edit `.env` from `.env.example` if needed (`HTSP_OUTPUT_DIR`, `HTSP_QUANTUM_BACKEND`, etc. — see [configuration.md](configuration.md)).

---

## 2. Generate instances

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode generate
```

Writes `output/raw/instances/n_<n>/instance_<k>.json` (or under `HTSP_OUTPUT_DIR`).

Optional: `--instance-config path/to/config.yaml` to override generation settings.

---

## 3. Run experiments (choose one or more)

All modes below expect instances on disk (step 2) unless your YAML only re-generates internally — for the stock preset YAMLs, run `generate` first.

**CUDA-Q (GPU):**

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode cudaq
# Optional:
.venv/bin/python -m experiments.main_experiment_workflow --mode cudaq --output /path/to/output
```

**Simulated annealing (CPU):**

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode sa
```

**Cirq preset (`n=5` TQUDO):**

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode cirq5
```

**Brute force (exact reference, small n):**

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode brute_force
```

**Custom YAML grid:**

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode experiment \
  --experiment-yaml src/experiments/experiment_cudaq_n5_qubo.yaml
```

**Optional calibration (writes under output):**

```bash
.venv/bin/python -m experiments.estimate_t0 --n-instances 5 --chi0 0.8
.venv/bin/python -m experiments.estimate_lambdas --formulation qubo --lambda-values 10,50,100,500,1000
```

---

## 4. Check feasibility (optional)

```bash
.venv/bin/python -m experiments.main_experiment_workflow --mode check_feasibility --check-solver cudaq
# or: cirq | simulated_annealing | brute_force
```

---

## 5. Analysis pipeline

```bash
make -f scripts/makefile analysis-all
# equivalent:
.venv/bin/python -m data_analysis.pipeline --output-root output
```

Produces `output/processed/*` and `output/images/*.png`.

---

## 6. View dashboard

```bash
make -f scripts/makefile results-web
```

Open: `http://localhost:8765/webpage_results/index.html`

---

## Variants

| Goal | Adjust |
|------|--------|
| Different output root | Set `HTSP_OUTPUT_DIR` or pass `--output` on workflow; use same path in `--output-root` for `data_analysis` and makefile (defaults to `output`). |
| Parallel instance solves | `cudaq_max_parallel_instances` / `cpu_max_parallel_instances` in YAML or env — [parallelism_and_vectorization.md](parallelism_and_vectorization.md), [development.md](development.md). |
| Lint / test | `make -f scripts/makefile lint`, `make -f scripts/makefile test` |

---

## Related docs

- [experiments_design_and_artifacts.md](experiments_design_and_artifacts.md)
- [analysis_and_figures.md](analysis_and_figures.md)
- [development.md](development.md)
