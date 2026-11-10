"""Episode loop (plan §4.10).

One episode = `(dataset, target, surrogate, acquisition, variant, n₀, seed)`.
Runs pool-based sequential BO for B iterations and returns the trajectory.

**The call order is what makes gate 10 hold.** Always call `acquisitions()`
first, with the same rng. That way base and TRACE see the identical posterior
samples, and only afterwards does TRACE do its extra attribution. Reversing
the order diverges rng consumption and the two variants see different TS
samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .acquisitions import ablation_seed, acquisitions, crn, target_utility
from .data import Pool
from .guards import OracleGuard
from .metrics import coherence_utility_corr
from .ranking import ranking_row
from .surrogate_q import surrogate_quality
from .policies import K_DEFAULT, pick_reference, select_base, select_trace
from .surrogates.gp import GPSurrogate
from .surrogates.rf import RFSurrogate
from .targets import Target, initial_design

SURROGATES = {"GP": GPSurrogate, "RF": RFSurrogate}


@dataclass
class Episode:
    dataset: str; target_q: float; surrogate: str; acquisition: str
    variant: str; n0: int; seed: int; budget: int = 30
    K: int = K_DEFAULT

    @property
    def shard_id(self) -> str:
        return (f"{self.dataset}|q{int(self.target_q*100):02d}|{self.surrogate}"
                f"|{self.acquisition}|{self.variant}|n{self.n0}|s{self.seed}")

    @property
    def crn_key(self) -> str:
        """G10 — RNG key. **Excludes the variant.**

        If the variant were included, base-TS and TRACE-TS would see different
        posterior samples, and the difference between the two methods could
        not be separated from RNG noise vs. reranking. This is exactly what
        gate 10 checks.
        """
        return (f"{self.dataset}|q{int(self.target_q*100):02d}|{self.surrogate}"
                f"|{self.acquisition}|n{self.n0}|s{self.seed}")


@dataclass
class Row:
    t: int; selected_idx: int; y_selected: float
    target_err: float; best_so_far_err: float
    ref_idx: int; n_actions_m: int; n_unique_coalitions: int; K_used: int
    coherence_selected: float; rank_base: int; rank_final: int
    is_hit: bool
    # ── §6.1 schema ────────────────────────────────────────────────────────
    corr_coherence_vs_delta_util: float   # ★ G9 diagnostic (key evidence for RQ4 in §6.5)
    n_surrogate_rows: int                 # App C — rows actually sent to the surrogate
    iter_wall_s: float                    # App C
    fit_wall_s: float
    attrib_wall_s: float
    rank_info: dict                       # one row of the ranking table (§6.1)
    surr_q: dict                          # one row of the surrogate_q table (§6.1)


def _make_surrogate(name: str, seed: int):
    if name in SURROGATES:
        return SURROGATES[name](seed=seed)
    if name == "TabPFN":
        from .surrogates.tabpfn import TabPFNSurrogate    # lazy import (GPU lane only)
        return TabPFNSurrogate(seed=seed)
    raise ValueError(f"unknown surrogate: {name}")


def run_episode(ep: Episode, pool: Pool, target: Target) -> list[Row]:
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, ep.n0, ep.seed))

    X, ys = pool.X, target.y_star
    hits = set(target.hit_idx.tolist())
    rows: list[Row] = []

    for t in range(1, ep.budget + 1):
        t_iter = time.perf_counter()
        obs = guard.observed
        y_obs = guard.y_of(obs)
        e_best = float(np.abs(y_obs - ys).min())

        t_fit = time.perf_counter()
        surr = _make_surrogate(ep.surrogate, ep.seed).fit(X[obs], y_obs)
        fit_s = time.perf_counter() - t_fit
        unobs = guard.unobserved(pool.n)
        if len(unobs) == 0:
            break

        rng = crn(ep.seed, t, ep.crn_key)   # ★ G10: variant excluded
        acq_all, Ysamp = acquisitions(surr, X[unobs], ys, e_best, rng,
                                      return_samples=True)
        acq = acq_all[ep.acquisition]

        t_att = time.perf_counter()
        if ep.variant == "base":
            sel = select_base(acq, unobs)
            n_unique = 0; ref_idx = -1; m = 0; corr = float("nan")
        else:
            ref_idx = pick_reference(guard, ys)
            ref = X[ref_idx]
            # ablation_seed is a **separate stream** from CRN. Consuming rng
            # here would diverge utility samples per variant and break gate 10.
            sel = select_trace(acq, unobs, ref, X, pool.meta,
                               lambda Z: target_utility(surr, Z, ys, rng),
                               K=ep.K, score=ep.variant,
                               ablation_seed=ablation_seed(ep.crn_key, t))
            att = sel.attribution
            # direct-utility has att None but utility_fn still uses K+1 rows.
            # Counting via n_unique would make App C report zero cost.
            n_unique = sel.util_rows
            m = int(att.m.max()) if att is not None else 0
            # ★ G9 — is coherence merely a monotone transform of ΔÛ? If the
            # correlation is near 1, rank-product just applied the acquisition
            # twice and the claim "explanation coherence adds separate
            # information" collapses (§6.5).
            corr = (coherence_utility_corr(att.coherence, att.delta_util)
                    if att is not None else float("nan"))

        attrib_s = time.perf_counter() - t_att

        # ── ranking table (§6.1, Fig 4) ───────────────────────────────────
        # Uses oracle y. Called **only after selection is done**, and reads
        # pool.y directly rather than through the policy path. Record-only;
        # it never feeds any decision in later iterations (spirit of gate 9).
        rinfo = ranking_row(sel.acq_top, sel.score,
                            pool.y[unobs[sel.shortlist]], ys,
                            sel.shortlist, len(unobs))
        # Reuses the samples the acquisition already drew (0 extra calls)
        sq = surrogate_quality(Ysamp, pool.y[unobs])

        y_new = guard.observe(sel.idx)
        err = abs(y_new - ys)
        rows.append(Row(
            t=t, selected_idx=sel.idx, y_selected=y_new, target_err=err,
            best_so_far_err=min(e_best, err), ref_idx=ref_idx,
            n_actions_m=m, n_unique_coalitions=n_unique, K_used=sel.K_used,
            coherence_selected=sel.coherence, rank_base=sel.rank_base,
            rank_final=sel.rank_final, is_hit=sel.idx in hits,
            corr_coherence_vs_delta_util=corr,
            n_surrogate_rows=len(unobs) + n_unique,
            iter_wall_s=time.perf_counter() - t_iter,
            fit_wall_s=fit_s, attrib_wall_s=attrib_s, rank_info=rinfo,
            surr_q=sq))

    return rows
