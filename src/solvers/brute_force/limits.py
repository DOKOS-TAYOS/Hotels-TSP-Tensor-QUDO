"""Hard limits for full brute-force enumeration (every assignment in the formulation space)."""

from __future__ import annotations

# QUBO: one binary per one-hot slot → n_vars = (n_cities - 1) ** 2 configurations: 2 ** n_vars.
QUBO_MAX_BINARY_VARS = 30

# TQUDO: sequences of length n_available over n_available symbols → n_available ** n_available configs.
TQUDO_MAX_N_AVAILABLE = 8
