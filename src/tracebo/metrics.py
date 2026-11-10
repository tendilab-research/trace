"""Metric computation (plan §6, G5·G6).

**This is the only place that calls `reveal_for_metrics()`.** If that symbol
appears in a policy module, gate 9 (static check) fails.

`ε` and `B` are not stored — they are computed post hoc from the trajectory
(§2.2). Since the policy depends on neither, sensitivity analysis comes
**for free** without rerunning.
"""

from __future__ import annotations

import numpy as np


def best_so_far(err: np.ndarray) -> np.ndarray:
    """Minimum target error up to t."""
    return np.minimum.accumulate(err)


def auc_error(err: np.ndarray, budget: int | None = None) -> float:
    """G6 — **primary metric**: Target-error AUC. **Lower is better.**

    The name evokes AUROC, but it is something else entirely. The curve's
    y-axis is "remaining error to the target", and the area under it is
    divided by B:

        (1/B) Σ_t  min_{s≤t} |y_s − y*|

    AUROC is 0.5–1 and higher is better; this is ≥ 0 and **lower** is better.
    Always report it as "Target-error AUC (lower is better)".

    Why Success@B is not the primary metric is arithmetic: under the spec's
    absolute ε, the random policy's Succ@20 is **0.64–1.00**, stuck at the
    ceiling (§1.4a), so no method is distinguishable. AUC measures "when and
    how close" over the whole horizon, so it does not saturate and changing
    the budget requires no rerun.
    """
    b = best_so_far(np.asarray(err, dtype=float))
    if budget is not None:
        b = b[:budget]
    return float(b.mean())


def success_at(hit: np.ndarray, budget: int) -> bool:
    """Succ@B — secondary metric. hit is the rank-based decision (G5)."""
    return bool(np.any(np.asarray(hit, dtype=bool)[:budget]))


def time_to_hit(hit: np.ndarray) -> float:
    """Time of first hit. `inf` if none — the median then handles censoring naturally."""
    idx = np.flatnonzero(np.asarray(hit, dtype=bool))
    return float(idx[0] + 1) if len(idx) else float("inf")


def simple_regret(err: np.ndarray) -> float:
    return float(best_so_far(np.asarray(err, dtype=float))[-1])


def coherence_utility_corr(coh: np.ndarray, dutil: np.ndarray) -> float:
    """G9 diagnostic — checks whether coherence is merely a monotone transform of ΔÛ.

    If the correlation is near 1, rank-product has effectively applied the
    acquisition twice and the claim "explanation coherence adds separate
    information" collapses. This is not a gate but a **diagnostic that
    governs how results are interpreted**, so it is recorded every iteration.
    """
    coh, dutil = np.asarray(coh, float), np.asarray(dutil, float)
    if len(coh) < 3 or coh.std() < 1e-12 or dutil.std() < 1e-12:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(coh, dutil).statistic)
