# Architecture Overview

## Purpose

This repository provides a reproducible research scaffold for a travel routing problem
(Hotel TSP) with precedence constraints, using Tensor-QUDO formulations and multiple
solver backends. Cost equations for both Tensor-QUDO and QUBO formulations are in
[formulations.md](formulations.md).

## High-level structure

- `src/config`: runtime settings loaded from `.env`.
- `src/instance_gen_process`: instance config, parsing, and generation.
- `src/solvers`: solver protocol plus backend stubs.
- `src/data_analysis`: raw-to-processed pipeline scaffold.
- `src/streamlit_app`: UI shell for reproducible runs.
- `src/utils`: shared logging and validation helpers.

## Data flow

1. Instance settings are loaded from `config.yaml`.
2. A `ProblemInstance` is generated and passed to a solver backend.
3. Solver outputs are stored as raw records in `output/raw`.
4. Analysis scripts curate records into benchmark-ready data in `output/processed`.
5. Paper figures are exported to `output/images`.

## Output policy

- Directory layout is versioned for consistency.
- Generated files are ignored by git.
- `.gitkeep` placeholders preserve expected folders in fresh clones.
