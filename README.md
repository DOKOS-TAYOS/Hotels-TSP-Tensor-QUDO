# Aircraft Loading Problem Tensor-QUDO

Research scaffold for combinatorial optimization on the Aircraft Loading Problem with
Tensor-QUDO formulations and multiple solver backends.

## Maintainers

- Adriano Lusso
- Alejandro Mata Ali

## Quickstart

### Linux / macOS

```bash
./install.sh
```

### Windows (CMD)

```bat
install.bat
```

Both setup scripts create `.venv`, install editable project dependencies, and default to
the `dev,ui,cirq` extras.
Installers validate `git` and `Python 3.12+` before running setup.

## Run common tasks

### Linux / macOS

```bash
make -f scripts/makefile lint
make -f scripts/makefile test
make -f scripts/makefile app
make -f scripts/makefile clean
```

### Windows (CMD)

```bat
scripts\make.bat lint
scripts\make.bat test
scripts\make.bat app
scripts\make.bat clean
```

## Project layout

```text
bin/                Bootstrap scripts
docs/               Project architecture and development notes
input/              Optional input datasets
output/             Local results (ignored in git except placeholders)
scripts/            Cross-platform task runners
src/
  config/           Runtime settings and environment loading
  data_analysis/    Raw -> processed analysis pipeline scaffolds
  instance_gen_process/
                    Instance configuration, loading, generation
  solvers/          Solver protocol + backend stubs
  streamlit_app/    Streamlit shell for reproducible experiments
  utils/            Shared validation and logging helpers
tests/              Smoke and contract tests
```

## Configuration

1. Copy `.env.example` to `.env`.
1. Adjust backend/output settings if needed.
1. Update `src/instance_gen_process/config.yaml` for instance generation.

## Output policy

- `output/raw`: intermediate experiment dumps.
- `output/processed`: curated benchmark datasets.
- `output/images`: figures for the paper.

These folders remain in the repo as placeholders, but generated contents are ignored.
