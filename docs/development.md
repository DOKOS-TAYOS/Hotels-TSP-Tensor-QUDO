# Development Guide

## Environment bootstrap

### Linux / macOS

```bash
./install.sh
```

### Windows (CMD)

```bat
install.bat
```

Default setup installs editable dependencies with extras `dev,ui,cirq`.

## Task runners

### Linux / macOS

```bash
make -f scripts/makefile setup
make -f scripts/makefile lint
make -f scripts/makefile test
make -f scripts/makefile app
```

### Windows (CMD)

```bat
scripts\make.bat setup
scripts\make.bat lint
scripts\make.bat test
scripts\make.bat app
```

## Code quality

- Target runtime: Python 3.12+.
- Linting: `ruff check .`
- Tests: `pytest`

## Branching and execution conventions

- Keep each experiment in a dedicated branch or folder under `output/raw`.
- Commit only code/config/docs. Do not commit generated artifacts.
- Record solver settings and seeds alongside every benchmark run.
