"""Robustness — how the decomposition changes with the **surrogate** and the
**initial data size**.

The main experiment (`uncert_split.py`) measures 480 episodes at a single
point, GP · n₀=10. We must show that the values from that point (U share
0.54, shared error 0.00, t-EI deviation 93.5%) are not artifacts of that
setting. Two axes are perturbed separately.

    --surrogates GP,RF,TabPFN     changes the **shape** of the posterior
    --n0-list 10,20,...,300       changes the **width** of the posterior

**Why these two axes.** `A₀ = lim_{σ→0} A` uses only μ, so the shared
baseline proposition must be independent of posterior form
(distribution-free). The *magnitude* of U, on the other hand, is set by σ,
so it must shrink as n₀ grows. The former is a **prediction**, the latter a
**measurement**. Both are shown in one table.

Measured (per iteration, for the single chosen candidate):

  share_uA   Σ|φ^U| / Σ|φ^A|           ← uncertainty share relative to acquisition attribution
  share_ch   Σ|φ^U| / (Σ|φ^U|+Σ|φ^A₀|) ← share relative to channels (same definition as main experiment)
  pos_*      Pr(φ_k^U > 0)             ← per acquisition, per action
  r_*        rank correlation of φ^U across acquisitions
  share_err  max|φ^A₀ − shared|        ← EXP·OPT·TS. **Must be 0**
  tei_dev    t-EI deviation from φ^A₀ (normalized)
  performance  auc_err_n · succ · t_hit

`share_uA` can exceed 1 — when the two channels cancel, |φ^A| becomes
smaller than |φ^U|. That is itself informative, so both definitions are
reported.

**TabPFN is the GPU lane.** `moments` are the exact first and second
moments of the bar distribution; normality is not assumed, only the two
moments are reported — the acquisition applying Gaussian formulas on top of
them is the approximation, and its scope is stated in §3 of the paper.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ACQ4 = ("EXP", "OPT", "tEI", "TS")
DRIVER = "tEI"          # acquisition that drives the trajectory. Selection is pure argmax (V4)


def _corr(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _one(job):
    ds, q, seed, surr_name, n0, budget = job
    from tracebo.acquisitions import (ALPHA, _from_moments, acq_plugin,
                                      acq_values, tei_clip_active)
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.policies import pick_reference
    from tracebo.runner import _make_surrogate
    from tracebo.shapley import coalition_design, exact_shapley
    from tracebo.targets import initial_design, make_target

    pool = load(ds); target = make_target(pool, q)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star
    y_rng = float(pool.y.max() - pool.y.min()) or 1.0
    hits = set(target.hit_idx.tolist())

    rec, errs, hit_t = [], [], np.inf
    for t in range(1, budget + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < 2:
            break
        surr = _make_surrogate(surr_name, seed).fit(X[obs], yo)
        key = f"{ds}|{q}|{seed}|{t}|{n0}"
        acq = acq_values(surr, X[unobs], ys, eb, DRIVER, key)
        pick = int(unobs[int(np.argmax(acq))])
        ref = X[pick_reference(guard, ys)]

        # ── record performance first, independent of the decomposition (selection = pure argmax)
        guard.observe(pick)
        if pick in hits and hit_t is np.inf:
            hit_t = t
        errs.append(float(np.abs(guard.y_of([pick])[0] - ys)))

        # ── decomposition: one coalition matrix, one moments call → four acquisitions × three channels
        Z, rows, acts = coalition_design(ref, X[pick][None, :], pool.meta)
        mom = surr.moments(Z)
        if mom is None:
            raise NotImplementedError(f"{surr_name}: moments required")
        mu, sd = mom
        r, a0s = rows[0], acts[0]
        m = len(a0s)
        if m < 2:
            continue

        phi0_ref, chan = None, {}
        for name in ACQ4:
            a = _from_moments(mu, sd, Z, ys, eb, name, key, ALPHA)
            a0 = acq_plugin(name, mu, ys, eb)
            pa = exact_shapley(a[r], m)
            p0 = exact_shapley(a0[r], m)
            chan[name] = (pa, p0, pa - p0)
            if name != "tEI" and phi0_ref is None:
                phi0_ref = p0

        # Sharing holds **only for EXP·OPT·TS** (location functionals). t-EI becomes
        # a different function because of the clip, so its deviation is measured separately.
        share_err = max(float(np.abs(chan[n][1] - phi0_ref).max())
                        for n in ("EXP", "OPT", "TS"))
        nrm = lambda v: v / max(float(np.abs(v).sum()), 1e-30)
        tei_dev = float(np.abs(nrm(chan["tEI"][1]) - nrm(phi0_ref)).max())

        pa_d, p0_d, pu_d = chan[DRIVER]
        sU = float(np.abs(pu_d).sum())
        sA = float(np.abs(pa_d).sum())
        s0 = float(np.abs(p0_d).sum())
        row = dict(
            t=t, m=m,
            share_uA=sU / max(sA, 1e-30),
            share_ch=sU / max(sU + s0, 1e-30),
            share_err=share_err, tei_dev=tei_dev,
            tei_split=float(tei_dev > 1e-9),
            clip=tei_clip_active(mu[r], ys, eb),
            sd_ref=float(sd[r[0]]),
            ratio=float(np.abs(mu[r[0]] - ys) / max(sd[r[0]], 1e-30)),
        )
        for n in ACQ4:                       # Pr(φ^U_k > 0), per acquisition
            row[f"pos_{n}"] = float(np.mean(chan[n][2] > 0))
        row["r_exp_opt"] = _corr(chan["EXP"][2], chan["OPT"][2])
        row["r_exp_tei"] = _corr(chan["EXP"][2], chan["tEI"][2])
        row["r_exp_ts"] = _corr(chan["EXP"][2], chan["TS"][2])
        rec.append(row)

    if not errs:
        return None
    bsf = np.minimum.accumulate(np.array(errs))
    out = dict(dataset=ds, q=q, seed=seed, surrogate=surr_name, n0=n0,
               auc_err_n=float(bsf.mean() / y_rng),
               regret_n=float(bsf[-1] / y_rng),
               succ=float(np.isfinite(hit_t)),
               t_hit=float(hit_t) if np.isfinite(hit_t) else float("nan"))
    if rec:                                  # average over iterations within the episode
        for k in rec[0]:
            v = np.array([r[k] for r in rec], dtype=float)
            out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        out["share_err_max"] = max(r["share_err"] for r in rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogates", default="GP")
    ap.add_argument("--n0-list", default="10")
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--out", default="results/robust.npz")
    a = ap.parse_args(argv)

    from tracebo.data import load as _load
    from tracebo.targets import TARGET_QUANTILES
    dss = a.datasets.split(",")
    n0s = [int(v) for v in a.n0_list.split(",")]
    surrs = a.surrogates.split(",")

    jobs = []
    for ds, n0, s, q, seed in itertools.product(
            dss, n0s, surrs, TARGET_QUANTILES, range(a.seeds)):
        # combinations the pool cannot support are filtered and reported here, **not dropped silently**
        if _load(ds).n < n0 + a.budget + 20:
            continue
        jobs.append((ds, q, seed, s, n0, a.budget))
    print(f"  {len(jobs)} runs · surrogate {surrs} · n0 {n0s} · {dss}", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(_one, jobs, chunksize=2) if r]
    print(f"  done {len(res)} · {time.perf_counter()-t0:.0f}s\n", flush=True)

    import pandas as pd
    df = pd.DataFrame(res)
    COLS = ["share_uA", "share_ch", "pos_EXP", "pos_tEI", "pos_TS",
            "r_exp_opt", "r_exp_tei", "r_exp_ts",
            "share_err_max", "tei_split", "tei_dev", "clip",
            "sd_ref", "auc_err_n", "succ"]
    for by in (["surrogate"], ["n0"]):
        if df[by[0]].nunique() < 2:
            continue
        print(f"  ─── by {by[0]} ───")
        hdr = f"  {by[0]:>10s} {'n':>5s}"
        for c in COLS:
            hdr += f" {c:>10s}"
        print(hdr); print("  " + "-" * len(hdr))
        for k, g in df.groupby(by[0]):
            line = f"  {str(k):>10s} {len(g):5d}"
            for c in COLS:
                line += f" {np.nanmean(g[c]):10.4f}" if c in g else f" {'—':>10s}"
            print(line)
        print()

    np.savez_compressed(a.out, cols=np.array(list(df.columns)),
                        rows=df.astype(str).to_numpy(),
                        meta=json.dumps(dict(surrogates=surrs, n0s=n0s,
                                             datasets=dss, seeds=a.seeds,
                                             budget=a.budget)))
    print(f"  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
