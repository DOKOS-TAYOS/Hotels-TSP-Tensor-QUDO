"""Naive-loop oracle vs vectorised ``calculate_tqudo_cost`` (detects indexing bugs)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from instance_gen_process.models import ProblemTQUDO
from oracles import tqudo_cost_naive_loops
from utils.costs import calculate_tqudo_cost


def _duplicate_sequences(n: int) -> list[list[int]]:
    """A few non-permutation sequences (duplicate cities) for penalty coverage."""
    out: list[list[int]] = []
    if n >= 2:
        out.append([0] * n)
        out.append([0, 0] + [1] * (n - 2) if n > 2 else [0, 0])
    if n >= 3:
        out.append([0, 1, 0] + [2] * (n - 3) if n > 3 else [0, 1, 0])
    return out


@pytest.mark.parametrize("n_qudits", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_tqudo_naive_matches_vectorised_random(n_qudits: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    d = n_qudits
    Etab = rng.standard_normal((n_qudits, d, d))
    Ett = rng.standard_normal((n_qudits, n_qudits, d, d))
    es = float(rng.uniform(0.5, 2.0))
    problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ett, energy_scale=es)

    seqs: list[np.ndarray] = [
        np.array(p, dtype=np.int64) for p in itertools.permutations(range(d))
    ]
    for extra in _duplicate_sequences(n_qudits):
        if len(extra) == n_qudits:
            seqs.append(np.array(extra, dtype=np.int64))

    for seq in seqs:
        expected = tqudo_cost_naive_loops(Etab, Ett, seq, es)
        got = calculate_tqudo_cost(problem, seq)
        assert got == pytest.approx(expected), f"seq={seq.tolist()} naive={expected} got={got}"
