"""Action-coalition Shapley and action coherence (spec §6.5, plan §4.6).

The core design is the **coalition design**: collect every subset recipe of
every candidate into one matrix first, and call the surrogate **exactly once
per iteration**. Calling per candidate would mean K=256 candidates × 95.7
coalitions on average = 24,500 separate calls, which alone exhausts the
budget on TabPFN.

The same recipe recurs across subsets of many candidates (especially the
empty set = reference recipe), so deduplication shrinks the row count further.

Byproducts come for free: a single `U` yields every ablation quantity
(singleton ψ, ΔÛ, random/opposite coherence). This is why Tier B uses no
extra GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from .actions import Action, apply, decompose
from .meta import RecipeMeta

EPS = 1e-12
KEY_DECIMALS = 12


@dataclass
class Attribution:
    coherence: np.ndarray          # (K,)   c(x) ∈ [0,1]
    phi: list[np.ndarray]          # K entries, each (m_k,)
    psi: list[np.ndarray]          # K entries, each (m_k,)  singleton effect v({k})−v(∅)
    delta_util: np.ndarray         # (K,)   v(A)−v(∅)
    m: np.ndarray                  # (K,)   number of actions
    n_unique: int                  # rows actually sent to the surrogate
    actions: list                  # K entries, each list[Action]. Needed by V2
                                   # (action-prune) to actually apply subsets —
                                   # discarded while only scalar summaries were used.
    banzhaf: list                  # K entries, each (m_k,). Comes from the same v
                                   # with **zero extra surrogate calls**. The Shapley
                                   # weights s!(m−s−1)!/m! put extreme mass on
                                   # |S|=0 and |S|=m−1, which are exactly where the
                                   # surrogate extrapolates most. Banzhaf weights
                                   # uniformly, giving the largest safety margin
                                   # under noise (Wang & Jia, Data Banzhaf,
                                   # AISTATS 2023). The cost is losing
                                   # efficiency: Σβ ≠ ΔÛ.


def _key(z: np.ndarray) -> tuple:
    return tuple(np.round(z, KEY_DECIMALS))


def _bits(mask: int, m: int) -> list[int]:
    return [b for b in range(m) if mask >> b & 1]


def coalition_design(ref: np.ndarray, cands: np.ndarray, meta: RecipeMeta
                     ) -> tuple[np.ndarray, list[np.ndarray], list[list[Action]]]:
    """Collect all coalition recipes of all candidates, deduplicated, into one `Z`.

    Returns:
      Z     — (n_unique, d) surrogate input matrix
      rows  — per candidate, an int array (2^m_k,). `rows[k][mask]` = row index in Z
      acts  — per candidate, the action list (used for labeling and ablation)
    """
    key2row: dict[tuple, int] = {}
    Z: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    acts_all: list[list[Action]] = []

    for x in cands:
        acts = decompose(ref, x, meta)
        m = len(acts)
        acts_all.append(acts)
        full = (1 << m) - 1
        r = np.empty(1 << m, dtype=np.int64)
        for mask in range(1 << m):
            # The two endpoints are **by definition** the reference recipe and
            # the candidate itself. Going through `apply` leaves a 1e-16
            # floating-point reconstruction error, and the TS coordinate hash
            # (`_pointwise_normal`) is not continuous: once that error crosses
            # a rounding boundary, z changes wholesale — identity error reached
            # 7.5e-3 on the two oer datasets. Using the originals removes it.
            if mask == 0:
                z = ref
            elif mask == full:
                z = x
            else:
                z = apply(ref, [acts[b] for b in _bits(mask, m)], meta)
            k = _key(z)
            idx = key2row.get(k)
            if idx is None:
                idx = key2row[k] = len(Z)
                Z.append(z)
            r[mask] = idx
        rows.append(r)

    return np.asarray(Z, dtype=np.float64), rows, acts_all


def _shapley_weights(m: int) -> np.ndarray:
    """Weight s!(m−s−1)!/m! for coalitions with |S|=s."""
    return np.array([factorial(s) * factorial(m - s - 1) / factorial(m)
                     for s in range(m)], dtype=np.float64)


def exact_shapley(v: np.ndarray, m: int) -> np.ndarray:
    """Exact Shapley values from 2^m coalition utilities. Assumes m ≤ 10.

    OER's m_max=9 → 512 coalitions. **Cheaper** than the permutation
    approximation (128×(m+1)=1280) and exact. The crossover is m=11 — which is
    why spec §6.5's m>8 approximation rule is not used for the main results
    (§8 decision 2).
    """
    masks = np.arange(1 << m)
    sizes = np.bitwise_count(masks) if hasattr(np, "bitwise_count") else \
        np.array([bin(x).count("1") for x in masks])
    w = _shapley_weights(m)
    phi = np.empty(m, dtype=np.float64)
    for k in range(m):
        sel = masks[(masks >> k & 1) == 0]           # coalitions not containing k
        phi[k] = np.sum(w[sizes[sel]] * (v[sel | (1 << k)] - v[sel]))
    return phi


def exact_banzhaf(v: np.ndarray, m: int) -> np.ndarray:
    """β_k = 2^{-(m-1)} · Σ_{S∌k} [v(S∪k) − v(S)]  — uniformly weighted marginal contribution.

    Takes the **same v** as `exact_shapley`, so this value is free.
    Efficiency does not hold (Σβ ≠ ΔÛ), so gate 3 applies to φ only.
    """
    masks = np.arange(1 << m)
    beta = np.empty(m, dtype=np.float64)
    w = 2.0 ** (-(m - 1))
    for k in range(m):
        sel = masks[(masks >> k & 1) == 0]
        beta[k] = w * np.sum(v[sel | (1 << k)] - v[sel])
    return beta


def coherence(phi: np.ndarray) -> float:
    """c = P / (P + N + ε).

    If all |φ| are 0 (no contribution at all) return 0.5 — treating 0/0 as 1
    would give a 'meaningless candidate' the highest coherence. Checked by gate 5.
    """
    P = float(np.clip(phi, 0.0, None).sum())
    N = float(np.clip(-phi, 0.0, None).sum())
    return 0.5 if P + N < EPS else P / (P + N + EPS)


def attribute_batch(ref: np.ndarray, cands: np.ndarray, meta: RecipeMeta,
                    utility_fn) -> Attribution:
    """Compute attribution for a whole candidate batch with **one surrogate call**.

    `utility_fn(Z) -> (n_unique,)` is the target-oriented utility (§4.7).
    """
    Z, rows, acts_all = coalition_design(ref, cands, meta)
    U = np.asarray(utility_fn(Z), dtype=np.float64)      # ★ the only surrogate call
    assert U.shape == (len(Z),), f"utility_fn returned shape {U.shape} != {(len(Z),)}"

    K = len(cands)
    coh = np.empty(K)
    dutil = np.empty(K)
    ms = np.empty(K, dtype=np.int64)
    phis: list[np.ndarray] = []
    psis: list[np.ndarray] = []
    bzs: list[np.ndarray] = []

    for c, (r, acts) in enumerate(zip(rows, acts_all)):
        m = len(acts)
        ms[c] = m
        v = U[r]
        if m == 0:
            # The reference recipe itself. No actions, hence no contributions.
            phis.append(np.zeros(0)); psis.append(np.zeros(0))
            bzs.append(np.zeros(0))
            coh[c] = 0.5; dutil[c] = 0.0
            continue
        phi = exact_shapley(v, m)
        phis.append(phi)
        bzs.append(exact_banzhaf(v, m))
        psis.append(v[1 << np.arange(m)] - v[0])          # singleton effects
        dutil[c] = v[-1] - v[0]                           # ΔÛ = v(A) − v(∅)
        coh[c] = coherence(phi)
        # Gate 3 — efficiency: Σφ_k == v(A) − v(∅)
        assert abs(phi.sum() - dutil[c]) < 1e-8 * max(1.0, abs(dutil[c])), \
            f"Shapley efficiency violated: Σφ={phi.sum():.6g} vs ΔÛ={dutil[c]:.6g}"

    return Attribution(coherence=coh, phi=phis, psi=psis,
                       delta_util=dutil, m=ms, n_unique=len(Z),
                       actions=acts_all, banzhaf=bzs)
