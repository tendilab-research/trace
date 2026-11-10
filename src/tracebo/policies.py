"""Selection policies — base and TRACE reranking (plan §4.9, G15).

TRACE does not discard the acquisition score. It **reranks by rank-product**:

    `s(x) = q_A(x) · q_X(x)`

`q_A` is the percentile rank of the acquisition score, `q_X` the percentile
rank of action coherence. Being a product, **both must be high** to win — a
candidate that is promising and at the same time consistently explained.
The point is that there are no parameters. A weighted sum
`λ·q_A + (1−λ)·q_X` would require tuning λ, and the criticism that the tuning
produced the result could not be avoided.

**G15 top-K.** Attribution is computed only for the top `K = min(256, |U_t|)`
candidates. Running 2^m coalitions over the whole pool is infeasible. Gate 14
verifies empirically that this approximation does not change the argmax
(≥99% agreement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import apply
from .meta import RecipeMeta
from .shapley import Attribution, attribute_batch

K_DEFAULT = 256


def pct_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank ∈ (0,1). Ties get the average rank (gate 6).

    Uses `(r + 0.5)/n` so that 0 and 1 never occur. A `q_X = 0` would zero
    the product and wipe out the `q_A` information entirely.
    """
    n = len(v)
    if n == 0:
        return v.astype(float)
    order = np.argsort(v, kind="stable")
    r = np.empty(n, dtype=float)
    r[order] = np.arange(n, dtype=float)
    # average rank for ties
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return (r + 0.5) / n


def _argmax_tiebreak(s: np.ndarray, acq: np.ndarray, pos: np.ndarray) -> int:
    """Argmax of s. Ties broken by acquisition score, then by smaller pool index.

    Without a fixed tie-break, determinism (gate 7a) breaks.
    """
    best = np.flatnonzero(s >= s.max() - 1e-15)
    if len(best) > 1:
        a = acq[best]
        best = best[a >= a.max() - 1e-15]
    return int(best[np.argmin(pos[best])])


@dataclass
class Selection:
    idx: int                            # selected pool index
    attribution: Attribution | None     # TRACE family only
    rank_base: int                      # rank by acquisition score (0 = best)
    rank_final: int                     # rank after reranking
    coherence: float                    # coherence of the selected candidate
    K_used: int
    # ── for the ranking table (§6.1, Fig 4) ────────────────────────────────
    # Export the reranking scores and shortlist. This information is gone once
    # the iteration ends; without passing it here there is no way to compute
    # rank fidelity afterwards.
    shortlist: np.ndarray | None = None  # positions within the unobs array
    acq_top: np.ndarray | None = None    # acquisition scores of the shortlist
    score: np.ndarray | None = None      # s = q_A·q_X. None for base
    util_rows: int = 0                   # rows actually fed to utility_fn (App C)


def pick_reference(guard, y_star: float) -> int:
    """Reference recipe = the **observed** one closest to the target.

    Uses only `guard.y_of` — looking at an unobserved y is an immediate
    OracleLeak. Ties go to the **most recent** observation (so the reference
    updates as exploration proceeds).
    """
    obs = guard.observed
    err = np.abs(guard.y_of(obs) - y_star)
    best = np.flatnonzero(err == err.min())
    return int(obs[best[-1]])


def select_base(acq: np.ndarray, unobs: np.ndarray) -> Selection:
    """Acquisition argmax. The baseline for every comparison."""
    i = int(np.argmax(acq))
    return Selection(idx=int(unobs[i]), attribution=None, rank_base=0,
                     rank_final=0, coherence=float("nan"), K_used=0,
                     shortlist=np.arange(len(acq)), acq_top=acq, score=None)


def select_trace(acq: np.ndarray, unobs: np.ndarray, ref: np.ndarray,
                 X: np.ndarray, meta: RecipeMeta, utility_fn,
                 K: int = K_DEFAULT, score: str = "coherence",
                 ablation_seed: int = 0) -> Selection:
    """Rank-product reranking.

    `score` switches the ablation — all of them **reuse the same attribution
    result**, so the surrogate is not called again (why Tier B uses no extra
    GPU).
    """
    order = np.argsort(-acq, kind="stable")
    top = order[:min(K, len(order))]

    if score == "action-prune":
        # V2 (§8 decision 0b) — **the only path that actually uses per-action
        # information.**
        #
        # Build a recipe z* keeping only actions with φ_k > 0. z* is a **new
        # recipe**, not any pool candidate, and gate 2 guarantees the
        # composition constraint (sum preserved on every subset). A scalar
        # baseline only scores candidates and cannot **create** them, so it
        # cannot mimic this behavior.
        #
        # The selection projects onto the nearest **unobserved pool candidate**
        # to z* (in the pool-based setting a synthetic recipe cannot actually
        # be run).
        att = attribute_batch(ref, X[unobs[top]], meta, utility_fn)
        Zs = np.stack([apply(ref, [a for a, f in zip(acts, ph) if f > 0.0], meta)
                       for acts, ph in zip(att.actions, att.phi)])
        u_pruned = np.asarray(utility_fn(Zs), dtype=float)   # K extra rows
        raw = u_pruned
        util_rows = att.n_unique + len(Zs)
        qA = pct_rank(acq[top]); qX = pct_rank(raw)
        s = qA * qX
        j = _argmax_tiebreak(s, acq[top], top)
        pick = _nearest_unobserved(Zs[j], X, unobs, meta)
        return Selection(
            idx=int(pick), attribution=att, rank_base=int(j),
            rank_final=int(np.argsort(-s, kind="stable").tolist().index(j)),
            coherence=float(att.coherence[j]), K_used=len(top),
            shortlist=top, acq_top=acq[top], score=s, util_rows=util_rows)

    if score == "direct-utility":
        # ΔÛ = v(A) − v(∅) needs only **2 rows** per candidate. Running the
        # full coalition design (12,470 rows on average) is pure waste. This
        # is why plan §3.2 classifies only this ablation as light.
        att = None
        xs = np.vstack([ref[None, :], X[unobs[top]]])
        u = np.asarray(utility_fn(xs), dtype=float)
        raw = u[1:] - u[0]
        util_rows = len(xs)
    else:
        att = attribute_batch(ref, X[unobs[top]], meta, utility_fn)
        raw = _ablation_score(att, score, seed=ablation_seed)
        util_rows = att.n_unique

    qA = pct_rank(acq[top])
    qX = pct_rank(raw)
    s = qA * qX

    j = _argmax_tiebreak(s, acq[top], top)
    coh = float(att.coherence[j]) if att is not None else float("nan")
    # rank_base is the **rank by acquisition**. Since top is sorted by -acq,
    # that rank is j itself. Using top[j] would insert the position within the
    # unobs array, leaking a meaningless value into the schema's
    # base_rank_selected (source of Fig 4).
    return Selection(
        idx=int(unobs[top[j]]), attribution=att,
        rank_base=int(j), rank_final=int(np.argsort(-s, kind="stable").tolist().index(j)),
        coherence=coh, K_used=len(top),
        shortlist=top, acq_top=acq[top], score=s, util_rows=util_rows)


def _nearest_unobserved(z: np.ndarray, X: np.ndarray, unobs: np.ndarray,
                        meta: RecipeMeta) -> int:
    """Nearest unobserved pool candidate to `z`. Distance is **normalized by
    column range**.

    Unnormalized Euclidean distance on a pool like AgNP, which mixes
    composition (0–100) and process (uL/min), lets one large-scale column
    dominate the distance.
    """
    Xu = X[unobs]
    rng = X.max(axis=0) - X.min(axis=0)
    rng = np.where(rng > 1e-12, rng, 1.0)
    d = np.abs((Xu - z[None, :]) / rng[None, :]).sum(axis=1)
    return int(unobs[int(np.argmin(d))])


def _ablation_score(att: Attribution, kind: str, seed: int) -> np.ndarray:
    """Tier B ablations (§4.6 table). All zero-cost — reinterpretations of the attribution."""
    if kind == "coherence":
        return att.coherence
    # `coherence-mag` (V1) is not implemented. P·N are already **magnitude
    # sums**, so Σφ/Σ|φ| = (P−N)/(P+N) = 2c − 1, which ranks identically to
    # coherence under pct_rank (measured AUC agrees to 5 decimals). See §8
    # decision 0b.
    if kind == "opposite":                       # gate 11: must be worse than base
        return 1.0 - att.coherence
    if kind == "random":                         # gate 12: must be indistinguishable from base
        # The seed must differ per (episode, iteration). Previously len(top)
        # was passed, and since K=256 almost always, **every iteration,
        # episode and dataset reused a single fixed permutation**. That is not
        # a random control but one fixed fake score, and gate 12 hinges on
        # the luck of that permutation.
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.permutation(att.coherence)
    if kind == "singleton":
        return np.array([p.max() if len(p) else 0.0 for p in att.psi])
    raise ValueError(f"unknown ablation: {kind}")


def select_acq_explain(acq: np.ndarray, unobs: np.ndarray, ref: np.ndarray,
                       X: np.ndarray, meta: RecipeMeta, acq_fn) -> Selection:
    """V4 — **the acquisition selects, and the explanation is its exact decomposition.**

    The previous design pitted two signals against each other. The top K were
    chosen by `acq(x)` and reordered by `c(x)`, but `c` was a third score
    derived from a **different quantity** (the utility ΔÛ). Hence (1)
    performance could degrade — measured in 12/12 cells — and (2) the
    explanation did not explain why we actually chose. This is the point that
    felt "unprincipled".

    Here the **object** of Shapley changes:

        v(S) = acq(apply(ref, S))     ⟹   Σ_k φ_k = acq(x) − acq(ref)

    The efficiency identity (gate 3) becomes gate 16 as-is. Consequences:

    - **Selection** = pure acquisition selection. The trajectory matches the
      baseline exactly. Performance loss is not 'small' but **structurally 0**.
    - **Explanation** = exact action decomposition of the value that made the
      selection. Not an approximation.
    - **K disappears.** Selection needs no attribution, so all unobserved
      candidates are considered. Attribution runs on **the one chosen
      candidate** only — K·2^m rows per iteration become 2^m rows
      (measured 3,668 → ~100 rows, about 40x savings).

    `acq_fn(Z) -> (len(Z),)` must be **deterministic**
    (`acquisitions.acq_values`). A Monte Carlo acquisition would fluctuate
    across batches, degrading the identity to an approximation — and the
    point of this design would be lost.
    """
    j = int(np.argmax(acq))
    pick = int(unobs[j])
    att = attribute_batch(ref, X[pick][None, :], meta, acq_fn)
    return Selection(idx=pick, attribution=att, rank_base=j, rank_final=j,
                     coherence=float(att.coherence[0]), K_used=len(unobs),
                     shortlist=None, acq_top=None, score=None,
                     util_rows=att.n_unique)
