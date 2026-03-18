# Development Guide

This project targets **Linux only**. CUDA-Q and the install/task scripts require a Linux environment.

## Environment bootstrap

```bash
./install.sh
```

Default setup installs editable dependencies with extras `dev,ui,cudaq`. Use
`./install.sh dev,ui,cirq` if you want the Cirq backend instead.

## Task runners

```bash
make -f scripts/makefile setup
make -f scripts/makefile lint
make -f scripts/makefile test
make -f scripts/makefile app
```

## Code quality

- Target runtime: Python 3.12+.
- Linting: `ruff check .`
- Tests: `pytest`

## Branching and execution conventions

- Keep each experiment in a dedicated branch or folder under `output/raw`.
- Commit only code/config/docs. Do not commit generated artifacts.
- Record solver settings and seeds alongside every benchmark run.
