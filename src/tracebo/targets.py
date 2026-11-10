"""Target construction and initial design D₀ (plan §1.6-5~7, G5·G13).

This module fixes three things.

1. **y\\*** — the target value. Use the **actual measured value** in the pool
   closest to the quantile. A synthetic value can be unreachable in the pool.
2. **ε_y** — the hit tolerance. It is **rank-based** (G5).
3. **D₀** — the initial observations. Every method must receive the same D₀
   for the paired comparison to hold (gate 8), so the seed manifest is
   pre-generated.

**Why rank-based.** The spec's `ε = 5%(Q95−Q05)` is on an absolute scale, so
the number of qualifying candidates varies per dataset, and measured
Success@20 of the random policy is 0.64–1.00 — **stuck at the ceiling**
(§1.4). Defining the tolerance as "the k-th nearest value to the target"
fixes the number of hit candidates at exactly k regardless of dataset, so
difficulty is controlled by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Pool

TARGET_QUANTILES = (0.2, 0.5, 0.8)


@dataclass(frozen=True)
class Target:
    dataset: str
    q: float                # 0.2 | 0.5 | 0.8
    y_star: float           # target value (an actual measurement in the pool)
    star_idx: int           # pool index holding y_star
    eps: float              # rank-based tolerance (G5)
    k: int                  # number of candidates counted as hits
    hit_idx: np.ndarray     # (k,) hit candidate indices — evaluation only, never passed to policies

    @property
    def name(self) -> str:
        return f"{self.dataset}@q{int(self.q * 100):02d}"


def tolerance_k(n: int) -> int:
    """G5 — `k = max(3, ⌈0.02N⌉)`.

    The floor of 3 prevents k=2 on small pools (AutoAM N=100), where
    tie-breaking would dominate the result.
    """
    return max(3, int(np.ceil(0.02 * n)))


def make_target(pool: Pool, q: float) -> Target:
    """Build one target for quantile q."""
    y = pool.y
    n = len(y)
    q_val = float(np.quantile(y, q))
    star_idx = int(np.argmin(np.abs(y - q_val)))    # snap to an actual pool value
    y_star = float(y[star_idx])

    err = np.abs(y - y_star)
    k = tolerance_k(n)
    order = np.argsort(err, kind="stable")          # ties by index order — deterministic
    eps = float(err[order[k - 1]])                  # distance to the k-th nearest value
    hit_idx = np.flatnonzero(err <= eps)            # may exceed k because of ties

    return Target(dataset=pool.meta.name, q=q, y_star=y_star, star_idx=star_idx,
                  eps=eps, k=k, hit_idx=hit_idx)


def all_targets(pool: Pool) -> list[Target]:
    return [make_target(pool, q) for q in TARGET_QUANTILES]


# ══════════════════════════════════════════════════════════════════════════
#  Initial design D₀
# ══════════════════════════════════════════════════════════════════════════
MAX_REJECT = 200


def initial_design(pool: Pool, target: Target, n0: int, seed: int) -> np.ndarray:
    """Draw D₀. **Redraw if a hit candidate lands in D₀** (G13).

    The original rule in spec §8.3 excludes points with `|y−y*| ≤ ε` from D₀,
    but with an absolute-scale ε the rejection rate on P3HT q20 climbs to
    **92–94%**, rejection sampling effectively never runs, and the surviving
    D₀ are biased toward "initializations that happened to dodge the target"
    (§1.4b).

    With rank-based k the rejection probability is bounded at roughly
    `1−C(N−k,n₀)/C(N,n₀)` ≈ `n₀k/N` (~10% for n₀=5, k=0.02N). If it still
    fails, **raise** — silently using a D₀ containing a hit would make that
    seed succeed from the start.
    """
    n = pool.n
    assert n0 < n - len(target.hit_idx), \
        f"{target.name}: n0={n0} is too large for the pool (N={n}, hit={len(target.hit_idx)})"
    hits = set(target.hit_idx.tolist())
    rng = np.random.Generator(np.random.PCG64(_d0_seed(target, n0, seed)))

    for _ in range(MAX_REJECT):
        idx = rng.choice(n, size=n0, replace=False)
        if not (set(idx.tolist()) & hits):
            return np.sort(idx)

    # ── Regime where rejection sampling fails. As n₀ grows relative to N,
    #    P(no hit) ≈ (1 − n₀/N)^k goes to 0 **exponentially**.
    #    crossed_barrel (N=600, k=12): 0.0077 at n₀=200, 2.4e-4 at n₀=300.
    #
    #    But the distribution we want is "uniform over n₀-subsets containing
    #    no hit", which is **exactly** uniform sampling from the non-hit set.
    #    So this is not an approximation — the same distribution, drawn differently.
    #
    #    For small n₀ the loop above succeeds first, so **existing results do
    #    not change by a single bit** — this branch only replaces the old raise.
    free = np.setdiff1d(np.arange(n), target.hit_idx)
    if len(free) < n0:
        raise RuntimeError(
            f"{target.name}: n0={n0} > {len(free)} non-hit candidates (N={n}, "
            f"k={target.k}). This (dataset, n₀) combination is infeasible.")
    return np.sort(rng.choice(free, size=n0, replace=False))


def _d0_seed(target: Target, n0: int, seed: int) -> int:
    """RNG seed for D₀. **Does not include the variant** — so every method for
    the same (dataset, target, n0, seed) receives the identical D₀ (gate 8)."""
    key = f"{target.dataset}|{target.q}|{n0}|{seed}"
    return int(np.frombuffer(_blake(key), dtype=np.uint64)[0] >> 1)


def _blake(s: str) -> bytes:
    from hashlib import blake2b

    return blake2b(s.encode(), digest_size=8).digest()


def manifest(pool: Pool, n0_values=(5,), seeds=range(30)) -> dict:
    """Pre-generate the seed manifest (§1.6-7). Built once and saved before the
    run; every worker reads it. Drawing at runtime cannot guarantee gate 8."""
    out = {}
    for t in all_targets(pool):
        for n0 in n0_values:
            for s in seeds:
                out[(t.dataset, t.q, n0, s)] = initial_design(pool, t, n0, s)
    return out


if __name__ == "__main__":
    from .data import load
    from .meta import DATASETS

    print(f"{'target':<24}{'N':>6}{'k':>5}{'y*':>12}{'eps':>12}{'hit':>6}{'D0 reject%':>12}")
    for name in DATASETS:
        p = load(name)
        for t in all_targets(p):
            # Measured rejection rate: fraction of 200 draws at n0=5 that contain a hit
            rng = np.random.Generator(np.random.PCG64(0))
            hits = set(t.hit_idx.tolist())
            rej = sum(bool(set(rng.choice(p.n, 5, replace=False).tolist()) & hits)
                      for _ in range(200)) / 200
            print(f"{t.name:<24}{p.n:>6}{t.k:>5}{t.y_star:>12.4g}{t.eps:>12.4g}"
                  f"{len(t.hit_idx):>6}{rej * 100:>11.1f}%")
