"""V6 — weight the uncertainty channel **per action** (extension of plan §8).

V5 had a single scalar knob:

    A_λ(x) = A₀(x) + λ·U(x)

That can only ask "how much do we trust this candidate's uncertainty as a
whole". But the action space is wide, and within a single candidate some
changes are **recommended** because of uncertainty (φ_k^U > 0) while others
are **suppressed** (φ_k^U < 0). A single scalar cannot treat those two
differently.

Shapley efficiency gives `Σ_k φ_k^U = U(x) − U(ref)`, so instead of using
U(x) wholesale we can **weight term by term**:

    add:  s(x) = A₀(x) + Σ_k w(g_k)·φ_k^U(x)
    mul:  s(x) = −E(x)·exp(−Σ_k w(g_k)·φ̂_k^U(x)),   E = |μ−y*|

With `w ≡ λ` (constant), add reduces exactly to V5's A_λ (efficiency).
So add is a **strict generalization** of A_λ. mul is a different object —
uncertainty shrinks the error **proportionally rather than absolutely**, so
a candidate already sitting on the target is moved less by uncertainty.

**Multiplicative = additive inside the exponent.** The "multiply per action"
form the user asked for is `Π_k exp(w_k φ̂_k)`, which is `exp(Σ_k w_k φ̂_k)`.
So the multiplicative and additive forms are the same object; the only
difference is **what is multiplied** (−E, or nothing). Implementing these
two closes the family.

**w must be a function of the action *type*.** If `w` were free per
candidate, the argmax could be manipulated at will and it would no longer be
an acquisition function. Every grouping used here is a type that spans
candidates:

    scalar — one group (= V5)
    sign   — sign of φ_k^U (actions uncertainty pushes / actions it vetoes)
    kind   — comp (composition transfer) / proc (process variable)

**Candidate shortlist and its price.** We cannot run 2^m coalitions over the
whole pool. We use the **union** of the top-K by `A`, `A₀`, and `−E`.

**Exact at the vertices, approximate inside.** The argmax of w≡1 (the
original acquisition) and w≡0 (plug-in) are in the union by definition —
`verify_claims.py` gives 960/960 with score loss exactly 0. But the argmax
of `A_w = (1−w)A₀ + wA` **may lie in neither top-K at interior points
0<w<1.** In 1,440 measured runs the optimum was missed 7 times (0.5%), with
a score loss of at most 1.6e-03 relative to the score range and median 0.
Every comparison uses the same shortlist, so this is not a bias, but **the
approximation goes into the paper's limitations.**

**w>1 is not covered at all.** It extrapolates and leaves both top-Ks by a
wide margin, so the grid is restricted to [0,1]. Since all four acquisition
functions in V5 preferred the **less-exploratory** side, this restriction
does not cut off the region of interest.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pin BLAS to 1 thread. Must happen **before numpy is imported** to take effect.
# 8 workers × 4 BLAS threads = 32 threads thrashing on 4 performance cores gave
# load average 35.8 and 47% CPU per worker. With ≤40 GP training points there
# is nothing to gain from BLAS parallelism — only thread-switch overhead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ACQS = ("EXP", "OPT", "tEI", "TS")
K_SHORT = 16

# (scorer, grouping) → list of weight tuples. Tuple length = number of groups.
#
# **The diagonal is excluded.** Giving both groups the same value under
# `sign`/`kind` is exactly `add-scalar` (efficiency). Counting the same policy
# under two names only inflates the multiple-comparison burden. So only the
# off-diagonal combinations remain, and `add-scalar` represents the scalar line.
def _configs(which: str, grid: str = "core") -> list[tuple]:
    """`grid="core"` is 11 configs (local), `"full"` is 17 (192-core pod).

    core is a reduced set that trades grid density for power. full restores
    the off-diagonal combinations — Holm-corrected comparisons grow to 16, but
    the seed count grows to compensate. **The diagonal is in neither**: giving
    both `sign`/`kind` groups the same value is exactly `add-scalar`, and
    counting the same policy under two names only inflates the multiple
    comparisons.
    """
    full = grid == "full"
    cfg: list[tuple] = []
    if "add-scalar" in which:
        ws = (0.0, 0.25, 0.5, 0.75, 1.0) if full else (0.0, 0.5, 1.0)
        cfg += [("add", "scalar", (w,)) for w in ws]
    if "add-sign" in which:
        ws = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (0.5, 1.0))
        if full:
            ws += ((1.0, 0.25), (0.25, 1.0))
        cfg += [("add", "sign", w) for w in ws]
    if "add-kind" in which:
        ws = ((1.0, 0.0), (0.0, 1.0))
        if full:
            ws += ((1.0, 0.5), (0.5, 1.0))
        cfg += [("add", "kind", w) for w in ws]
    if "mul-scalar" in which:
        ws = (0.25, 0.5, 1.0) if full else (0.5, 1.0)
        cfg += [("mul", "scalar", (w,)) for w in ws]
    if "mul-sign" in which:
        cfg += [("mul", "sign", w) for w in ((1.0, 0.5), (0.5, 1.0))]
    return cfg


def _group_idx(grouping: str, phiU: np.ndarray, kinds: list[str]) -> np.ndarray:
    """Group index per action."""
    if grouping == "scalar":
        return np.zeros(len(phiU), dtype=np.int64)
    if grouping == "sign":
        return (phiU < 0).astype(np.int64)          # 0=push, 1=veto
    if grouping == "kind":
        return np.array([0 if k == "comp" else 1 for k in kinds], dtype=np.int64)
    raise ValueError(grouping)


def _one(job):
    ds, q, seed, acq, scorer, grouping, w, n0, budget, K = job
    from tracebo.acquisitions import ALPHA, _from_moments, _moments, acq_plugin
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.policies import pick_reference
    from tracebo.runner import _make_surrogate
    from tracebo.shapley import coalition_design, exact_shapley
    from tracebo.targets import initial_design, make_target

    W = np.asarray(w, dtype=np.float64)
    pool = load(ds); target = make_target(pool, q)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star
    y_rng = float(pool.y.max() - pool.y.min()) or 1.0
    hits = set(target.hit_idx.tolist())

    errs, hit_t, n_short = [], np.inf, []
    for t in range(1, budget + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < 2:
            break
        surr = _make_surrogate("GP", seed).fit(X[obs], yo)
        # Key without λ·w — every config sees the same TS sample (G10)
        key = f"{ds}|{q}|{seed}|{t}"
        Zc = X[unobs]
        mu, sd = _moments(surr, Zc)
        a = _from_moments(mu, sd, Zc, ys, eb, acq, key, ALPHA)
        a0 = acq_plugin(acq, mu, ys, eb)

        # ── shortlist = top-K(A) ∪ top-K(A₀) ∪ top-K(−E).
        #    Three layers so that all **three vertices** of the family are covered:
        #      add w≡1 → argmax A,  add w≡0 → argmax A₀,  mul w≡0 → argmax(−E).
        #    (tEI has A₀ = clip(e−E), so A₀ and −E rank differently.)
        #    The vertices (w=0,1) are exact, **the interior is approximate** —
        #    7 of 1,440 runs miss the optimum, score loss ≤1.6e-03 (`verify_claims.py`).
        #    **w>1 is not covered at all** (extrapolation), hence the grid is [0,1].
        k = min(K, len(unobs))
        top = lambda v: np.argpartition(-v, k - 1)[:k]
        sl = np.union1d(np.union1d(top(a), top(a0)), top(-np.abs(mu - ys)))
        n_short.append(len(sl))
        ref = X[pick_reference(guard, ys)]

        Z, rows, acts = coalition_design(ref, X[unobs[sl]], pool.meta)
        mu2, sd2 = _moments(surr, Z)
        aZ = _from_moments(mu2, sd2, Z, ys, eb, acq, key, ALPHA)
        a0Z = acq_plugin(acq, mu2, ys, eb)
        uZ = aZ - a0Z

        # Solve φ^U **once** per candidate — Shapley dominates the cost of this loop.
        sumU = np.empty(len(sl))                 # Σ_k w·φ_k^U (per candidate)
        tot = np.zeros(len(sl))                  # Σ_k |φ_k^U| — scale for mul
        for c, (r, ac) in enumerate(zip(rows, acts)):
            m = len(ac)
            if m == 0:
                sumU[c] = 0.0
                continue
            phiU = exact_shapley(uZ[r], m)
            g = _group_idx(grouping, phiU, [x.kind for x in ac])
            sumU[c] = float(np.dot(W[g], phiU))
            tot[c] = float(np.abs(phiU).sum())

        if scorer == "add":
            # A₀(x) + Σ w φ^U.  With w≡λ this = A₀ + λ(U(x)−U(ref)) = A_λ + const
            s = a0[sl] + sumU
        else:
            # −E·exp(−Σ w φ̂).  φ̂ is made dimensionless by a per-iteration robust scale.
            sc = float(np.median(tot[tot > 0])) if np.any(tot > 0) else 1.0
            E = np.abs(mu[sl] - ys)
            s = -E * np.exp(-np.clip(sumU / max(sc, 1e-30), -20.0, 20.0))

        pick = int(unobs[sl[int(np.argmax(s))]])
        guard.observe(pick)
        if pick in hits and hit_t is np.inf:
            hit_t = t
        errs.append(float(np.abs(guard.y_of([pick])[0] - ys)))

    if not errs:
        return None
    bsf = np.minimum.accumulate(np.array(errs))
    return dict(dataset=ds, q=q, seed=seed, acq=acq,
                cfg=f"{scorer}-{grouping}", w="/".join(f"{v:g}" for v in w),
                auc_err_n=float(bsf.mean() / y_rng),
                regret_n=float(bsf[-1] / y_rng),
                succ=float(np.isfinite(hit_t)), t_hit=float(hit_t),
                n_short=float(np.mean(n_short)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--K", type=int, default=K_SHORT)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--acqs", default=",".join(ACQS))
    ap.add_argument("--configs",
                    default="add-scalar,add-sign,add-kind,mul-scalar")
    ap.add_argument("--grid", default="core", choices=("core", "full"))
    ap.add_argument("--out", default="results/actionw.npz")
    a = ap.parse_args(argv)

    from tracebo.data import load as _load
    from tracebo.meta import SUITE_FULL
    from tracebo.targets import TARGET_QUANTILES
    dss = a.datasets.split(",") if a.datasets else list(SUITE_FULL)
    dss = [d for d in dss if _load(d).n >= a.n0 + a.budget + 20]
    acqs = a.acqs.split(",")
    cfgs = _configs(a.configs, a.grid)

    jobs = [(ds, q, s, acq, sc, gr, w, a.n0, a.budget, a.K)
            for ds, q, s, acq, (sc, gr, w) in itertools.product(
                dss, TARGET_QUANTILES, range(a.seeds), acqs, cfgs)]
    print(f"  {len(jobs)} runs ({len(dss)}ds × 3q × {a.seeds}seed × "
          f"{len(acqs)}acq × {len(cfgs)}cfg) · K={a.K}", flush=True)
    t0 = time.perf_counter()
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                res.append(r)
            if i % 100 == 0 or i == len(futs):
                el = time.perf_counter() - t0
                eta = el / i * (len(futs) - i)
                print(f"  {i}/{len(futs)}  {el:.0f}s elapsed · "
                      f"ETA {eta/60:.1f}min", flush=True)
    el = time.perf_counter() - t0
    print(f"  done {len(res)} · {el:.0f}s\n", flush=True)

    import pandas as pd
    from scipy.stats import wilcoxon
    df = pd.DataFrame(res)
    KEY = ["dataset", "q", "seed", "acq"]
    # Baseline = original acquisition = add-scalar w=1 (same ranking as A by efficiency)
    base = df[(df.cfg == "add-scalar") & (df.w == "1")].set_index(KEY)
    assert len(base), "no baseline (add-scalar w=1) — include add-scalar in --configs"

    print(f"  {'acq':5s} {'cfg':12s} {'w':>9s}  {'auc':>9s}  {'Δ vs base':>10s}"
          f"  {'p':>9s}  {'p_holm':>9s}  {'win':>6s}  {'Succ':>6s}")
    print("  " + "-" * 84)
    best = {}
    for acq in acqs:
        rows = []
        for (cfg, w), cur in df[df.acq == acq].groupby(["cfg", "w"], sort=False):
            j = cur.set_index(KEY).join(base, rsuffix="_b", how="inner")
            if not len(j):
                continue
            d = (j.auc_err_n - j.auc_err_n_b).to_numpy()
            is_base = (cfg == "add-scalar" and w == "1")
            try:
                p = 1.0 if is_base or not np.any(d != 0) else wilcoxon(d).pvalue
            except ValueError:
                p = float("nan")
            rows.append((float(j.auc_err_n.mean()), d.mean(), p,
                         float((d < 0).mean()), float(j.succ.mean()),
                         cfg, w, is_base))
        # Multiple comparisons. Within one acquisition, 16 configs are compared
        # against the baseline. Counting p<0.05 uncorrected, one will show up by
        # chance. Holm controls FWER and is less conservative than Bonferroni.
        cand = [r for r in rows if not r[7] and np.isfinite(r[2])]
        order = np.argsort([r[2] for r in cand])
        n = len(cand)
        holm = {}
        run = 0.0
        for rank, oi in enumerate(order):
            run = max(run, min(1.0, (n - rank) * cand[oi][2]))
            holm[(cand[oi][5], cand[oi][6])] = run
        for m, dm, p, win, sc, cfg, w, is_base in sorted(rows):
            ph = holm.get((cfg, w), float("nan"))
            tag = (" ← base" if is_base else
                   (" ★" if ph < 0.05 and dm < 0 else
                    (" (raw)" if p < 0.05 and dm < 0 else "")))
            print(f"  {acq:5s} {cfg:12s} {w:>9s}  {m:9.5f}  {dm:+10.5f}"
                  f"  {p:9.2e}  {ph:9.2e}  {win:6.3f}  {sc:6.3f}{tag}")
            if not is_base and dm < best.get(acq, (0.0,))[0]:
                best[acq] = (dm, cfg, w, p, ph)
        print()

    print("  Best config beating the baseline:")
    for acq in acqs:
        if acq in best:
            dm, cfg, w, p, ph = best[acq]
            print(f"    {acq:5s} {cfg}  w={w}  Δ={dm:+.5f}  "
                  f"p={p:.2e}  p_holm={ph:.2e}"
                  f"{'  ★ significant' if ph < 0.05 else '  (n.s. after correction)'}")
        else:
            print(f"    {acq:5s} none — negative result")

    np.savez_compressed(a.out, cols=np.array(list(df.columns)),
                        rows=df.astype(str).to_numpy(),
                        meta=json.dumps(dict(K=a.K, budget=a.budget,
                                             n0=a.n0, seeds=a.seeds)))
    print(f"\n  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
