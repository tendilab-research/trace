"""Performance of the new acquisition A_λ that V5 yields (plan §8).

With the decomposition, the uncertainty channel can be **separated and weighted**:

    A_λ(x) = A₀(x) + λ·U(x) = (1−λ)·A₀(x) + λ·A(x)

λ=1 is the original acquisition, λ=0 pure plug-in, λ<0 flips the valuation of uncertainty.

**Why this knob is new.** In target-oriented BO, `U_EXP ≤ 0` is forced by
Jensen because the utility `u = −|y−y*|` is **concave**. So EXP is
necessarily risk-averse with no way to adjust it. For tEI, `e_best` is a
given value, not a parameter. Nothing corresponding to UCB's β or EI's ξ
**exists** on this side — because the σ-dependent part could not be
separated before A₀ was isolated.

**Fairness.** λ changes only the selection rule. The initial design is
determined solely by `(pool, target, n0, seed)`, and TS randomness is a
coordinate hash (λ not included), so every λ sees the **same posterior
sample**. Hence differences come only from λ.

The primary metric is `auc_err_n` from plan §6 — the iteration mean of the
best-so-far target error divided by the y range (**lower is better**). λ=1
serves as the paired baseline.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ACQS = ("EXP", "OPT", "tEI", "TS")
LAMBDAS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _one(job):
    ds, q, seed, acq, lam, n0, budget = job
    from tracebo.acquisitions import ALPHA, _from_moments, _moments, acq_plugin
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.targets import initial_design, make_target

    pool = load(ds); target = make_target(pool, q)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star
    y_rng = float(pool.y.max() - pool.y.min()) or 1.0
    hits = set(target.hit_idx.tolist())

    from tracebo.runner import _make_surrogate
    errs, hit_t = [], np.inf
    for t in range(1, budget + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < 2:
            break
        surr = _make_surrogate("GP", seed).fit(X[obs], yo)
        # Key without λ — every λ sees the same TS sample (G10)
        key = f"{ds}|{q}|{seed}|{t}"
        Z = X[unobs]
        mu, sd = _moments(surr, Z)
        a = _from_moments(mu, sd, Z, ys, eb, acq, key, ALPHA)
        a0 = acq_plugin(acq, mu, ys, eb)
        pick = int(unobs[int(np.argmax(a0 + lam * (a - a0)))])
        guard.observe(pick)
        if pick in hits and hit_t is np.inf:
            hit_t = t
        errs.append(float(np.abs(guard.y_of([pick])[0] - ys)))

    if not errs:
        return None
    bsf = np.minimum.accumulate(np.array(errs))   # best-so-far target error
    return dict(dataset=ds, q=q, seed=seed, acq=acq, lam=lam,
                auc_err_n=float(bsf.mean() / y_rng),
                regret_n=float(bsf[-1] / y_rng),
                succ=float(np.isfinite(hit_t)),
                t_hit=float(hit_t))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--acqs", default=",".join(ACQS))
    ap.add_argument("--out", default="results/lambda.npz")
    a = ap.parse_args(argv)

    from tracebo.data import load as _load
    from tracebo.meta import SUITE_FULL
    from tracebo.targets import TARGET_QUANTILES
    dss = a.datasets.split(",") if a.datasets else list(SUITE_FULL)
    dss = [d for d in dss if _load(d).n >= a.n0 + a.budget + 20]
    acqs = a.acqs.split(",")

    jobs = [(ds, q, s, acq, lam, a.n0, a.budget)
            for ds, q, s, acq, lam in itertools.product(
                dss, TARGET_QUANTILES, range(a.seeds), acqs, LAMBDAS)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(_one, jobs, chunksize=8) if r]
    print(f"\n  GP · n0={a.n0} · budget={a.budget} · runs {len(res)} · "
          f"{time.perf_counter()-t0:.0f}s\n", flush=True)

    import pandas as pd
    from scipy.stats import wilcoxon
    df = pd.DataFrame(res)
    KEY = ["dataset", "q", "seed", "acq"]
    base = df[df.lam == 1.0].set_index(KEY)

    print(f"  {'acq':5s} {'λ':>5s}  {'auc_err_n':>10s}  {'Δ vs λ=1':>10s}"
          f"  {'p':>9s}  {'win':>6s}  {'Succ':>6s}")
    print("  " + "-" * 62)
    best = {}
    for acq in acqs:
        for lam in LAMBDAS:
            cur = df[(df.acq == acq) & (df.lam == lam)].set_index(KEY)
            j = cur.join(base, rsuffix="_b", how="inner")
            d = (j.auc_err_n - j.auc_err_n_b).to_numpy()
            m = float(j.auc_err_n.mean()); sc = float(j.succ.mean())
            if lam == 1.0:
                print(f"  {acq:5s} {lam:+5.2f}  {m:10.5f}  {'base':>10s}"
                      f"  {'':>9s}  {'':>6s}  {sc:6.3f}")
                continue
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            win = float((d < 0).mean())
            star = " ★" if (p < 0.05 and d.mean() < 0) else ""
            print(f"  {acq:5s} {lam:+5.2f}  {m:10.5f}  {d.mean():+10.5f}"
                  f"  {p:9.2e}  {win:6.3f}  {sc:6.3f}{star}")
            if d.mean() < best.get(acq, (0.0, None))[0]:
                best[acq] = (d.mean(), lam, p)
        print()

    print("  Best λ (by auc_err_n, lower is better):")
    for acq, (dm, lam, p) in best.items():
        print(f"    {acq:5s} λ*={lam:+.2f}  Δ={dm:+.5f}  p={p:.2e}")
    if not best:
        print("    No λ beats λ=1 — negative result.")

    np.savez_compressed(a.out, cols=np.array(list(df.columns)),
                        rows=df.astype(str).to_numpy())
    print(f"\n  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
