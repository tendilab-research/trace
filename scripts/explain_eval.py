"""Evaluate the explanation itself — not optimization performance (main text §3.3).

Three axes, and the order matters:

  E1 Exactness              does the identity hold numerically          ← sanity
  E2 Acquisition discrimination  does it discriminate on real data      ← core evidence
  E3 Synthetic recovery     does it recover a known ground truth        ← validity defense

Three-step structure from a reviewer's viewpoint: *the identity holds → it
recovers a controlled ground truth → it finds meaningful differences on real
materials data.*

**E1 has a trap.** `uncert_split.py` constructs `φ^U := φ^A − φ^{A₀}`, so the
residual measured there is identically 0 and tests nothing. Here the three
channels are solved **independently** — `exact_shapley` is run separately on
the three value vectors `A`, `A₀`, `U=A−A₀`, and we check that they mesh.
Only then does the number mean something.

**Action validity is not measured.** The paired transfer `δ_{j→i}` moves −δ
and +δ as a pair, so the composition sum is preserved **for any subset
applied** — a design guarantee, not a measured quantity. Still, the 0%
violation rate is confirmed in one line.
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

ACQ = ("EXP", "OPT", "tEI", "TS")
PAIRS = tuple(itertools.combinations(ACQ, 2))
DRIVER = "tEI"
SHARED = ("EXP", "OPT", "TS")      # location functionals of the shared utility — share A₀


# ══════════════════════════════════════════════════════════ E1 + E2 (real data)
def _episode(job):
    ds, q, seed, n0, budget = job
    from tracebo.acquisitions import ALPHA, _from_moments, acq_plugin, acq_values
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.policies import pick_reference
    from tracebo.runner import _make_surrogate
    from tracebo.shapley import coalition_design, exact_shapley
    from tracebo.targets import initial_design, make_target
    from scipy.stats import spearmanr

    pool = load(ds); target = make_target(pool, q)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star

    rows = []
    for t in range(1, budget + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < 2:
            break
        surr = _make_surrogate("GP", seed).fit(X[obs], yo)
        key = f"{ds}|{q}|{seed}|{t}"
        pick = int(unobs[int(np.argmax(
            acq_values(surr, X[unobs], ys, eb, DRIVER, key)))])
        ref = X[pick_reference(guard, ys)]
        guard.observe(pick)

        Z, rr, acts = coalition_design(ref, X[pick][None, :], pool.meta)
        mu, sd = surr.moments(Z)
        r, m = rr[0], len(acts[0])
        if m < 2:
            continue

        # ── **Same D_t · same x · same action set**. Only the acquisition changes.
        ch = {}
        e1 = 0.0
        for nm in ACQ:
            a = _from_moments(mu, sd, Z, ys, eb, nm, key, ALPHA)
            a0 = acq_plugin(nm, mu, ys, eb)
            # Solve the three channels **independently** (the point of E1)
            pa = exact_shapley(a[r], m)
            p0 = exact_shapley(a0[r], m)
            pu = exact_shapley((a - a0)[r], m)
            ch[nm] = (pa, p0, pu)
            sc = max(1e-30, float(np.abs(pa).max()))
            e1 = max(e1, float(np.abs(pa - p0 - pu).max()) / sc)

        rec = dict(dataset=ds, q=q, seed=seed, t=t, m=m, e1_rel=e1)

        # ── E2-a: is the value channel really shared (EXP·OPT·TS). t-EI separately.
        base = ch["EXP"][1]
        rec["dvalue_shared"] = max(float(np.abs(ch[n][1] - base).max())
                                   for n in SHARED)
        nrm = lambda v: v / max(float(np.abs(v).sum()), 1e-30)
        rec["dvalue_tei"] = float(np.abs(nrm(ch["tEI"][1]) - nrm(base)).max())

        # ── E2-b: rank correlation + sign disagreement of the uncertainty channel (all 6 pairs)
        for a1, a2 in PAIRS:
            u1, u2 = ch[a1][2], ch[a2][2]
            ok = np.isfinite(u1) & np.isfinite(u2)
            rho = float("nan")
            if ok.sum() >= 3 and u1[ok].std() > 1e-18 and u2[ok].std() > 1e-18:
                rho = float(spearmanr(u1[ok], u2[ok]).correlation)
            rec[f"rho_{a1}_{a2}"] = rho
            # Sign disagreement — exclude numerical noise near 0
            s1, s2 = np.sign(u1), np.sign(u2)
            live = (np.abs(u1) > 1e-14 * max(1.0, np.abs(u1).max())) & \
                   (np.abs(u2) > 1e-14 * max(1.0, np.abs(u2).max()))
            rec[f"sgn_{a1}_{a2}"] = (float(np.mean(s1[live] != s2[live]))
                                     if live.any() else float("nan"))
        rows.append(rec)
    return rows


# ══════════════════════════════════════════════════════════ E3 (synthetic ground truth)
def _synth_meta(d: int):
    """Recipe with d process variables only. No composition, so action = d coordinate substitutions."""
    from tracebo.meta import RecipeMeta
    return RecipeMeta(
        name="synth", family="synthetic",
        columns=tuple(f"v{i}" for i in range(d)),
        comp_idx=np.zeros(0, dtype=np.int64),
        proc_idx=np.arange(d, dtype=np.int64),
        comp_total=0.0, y_name="y", y_direction="max")


def _synth_case(kind: str, y_star=2.0, e_best=1.0):
    """`kind='split'`  a0,a1 move only μ / a2,a3 move only σ.
       `kind='sigma'`  **all four actions move only σ** (μ fixed).

    The second is the stronger control. If μ does not move at all, `A₀ = u(μ)`
    is constant, so `φ^{A₀} = 0` must hold **for any nonlinear acquisition**.
    This blocks the objection "it only works in easy cases like UCB that are
    already split into two terms".
    """
    from tracebo.acquisitions import ALPHA, _from_moments, acq_plugin
    from tracebo.shapley import coalition_design, exact_shapley

    d = 4
    meta = _synth_meta(d)
    ref = np.zeros(d)
    cand = np.ones(d)
    Z, rr, acts = coalition_design(ref, cand[None, :], meta)
    r, m = rr[0], len(acts[0])
    assert m == d, f"got {m} actions (expected {d})"

    if kind == "split":
        mu = 1.0 + 0.7 * Z[:, 0] + 0.4 * Z[:, 1]        # coords 0,1 only
        sd = 0.30 + 0.50 * Z[:, 2] + 0.25 * Z[:, 3]     # coords 2,3 only
        val_k, unc_k = [0, 1], [2, 3]
    else:
        mu = np.full(len(Z), 1.0)                        # **constant**
        sd = 0.30 + 0.40 * Z[:, 0] + 0.30 * Z[:, 1] \
                  + 0.20 * Z[:, 2] + 0.10 * Z[:, 3]
        val_k, unc_k = [], [0, 1, 2, 3]

    out = {}
    for nm in ("CB",) + ACQ:
        a = _from_moments(mu, sd, Z, y_star, e_best, nm, "synth", ALPHA)
        a0 = acq_plugin(nm, mu, y_star, e_best)
        p0 = exact_shapley(a0[r], m)
        pu = exact_shapley((a - a0)[r], m)
        tot = np.abs(p0) + np.abs(pu)
        leak = lambda idx, num: (float(np.mean(np.abs(num)[idx] /
                                               np.maximum(tot[idx], 1e-300)))
                                 if len(idx) else float("nan"))
        out[nm] = dict(
            L_U=leak(unc_k, p0),      # do σ-only actions leak into value
            L_V=leak(val_k, pu),      # do μ-only actions leak into uncertainty
            phi0_unc_max=float(np.abs(p0[unc_k]).max()) if unc_k else float("nan"),
            phiU_val_max=float(np.abs(pu[val_k]).max()) if val_k else float("nan"),
        )
    return out


def _constraint_check(n=200):
    """Do **all coalitions** on composition datasets respect the sum constraint (design check)."""
    from tracebo.data import load
    from tracebo.actions import apply, decompose, renorm
    from tracebo.shapley import _bits
    bad = tot = 0
    worst = 0.0
    for ds in ("p3ht", "agnp", "oer_plate_3496", "photo_pce10"):
        pool = load(ds); meta = pool.meta
        if not meta.n_comp:
            continue
        rng = np.random.default_rng(0)
        for _ in range(n):
            i, j = rng.integers(0, pool.n, 2)
            ref, cand = pool.X[i], pool.X[j]
            acts = decompose(ref, cand, meta)
            if not (1 <= len(acts) <= 12):
                continue
            target = float(renorm(ref[meta.comp_idx], meta.comp_total).sum())
            for mask in range(1 << len(acts)):
                z = apply(ref, [acts[b] for b in _bits(mask, len(acts))], meta)
                c = z[meta.comp_idx]
                err = abs(float(c.sum()) - target)
                worst = max(worst, err)
                tot += 1
                if err > 1e-9 or (c < -1e-12).any():
                    bad += 1
    return bad, tot, worst


# ══════════════════════════════════════════════════════════════════════ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--out", default="results/explain_eval.npz")
    a = ap.parse_args(argv)

    # ── E3 first: cheap, and if it fails there is no reason to look at the rest ──
    print("\n═══ E3  synthetic recovery (channel leakage) ═══")
    for kind, note in (("split", "a0,a1 = μ only · a2,a3 = σ only"),
                       ("sigma", "all four actions σ only (μ constant) — strong control")):
        res = _synth_case(kind)
        print(f"\n  [{kind}] {note}")
        print(f"    {'acq':5s} {'L_U (σ→value leak)':>18s} {'L_V (μ→uncert leak)':>22s}"
              f" {'max|φ^A₀| on σ-only':>22s}")
        for nm, v in res.items():
            lu = f"{v['L_U']:.3e}" if np.isfinite(v["L_U"]) else "—"
            lv = f"{v['L_V']:.3e}" if np.isfinite(v["L_V"]) else "—"
            pm = f"{v['phi0_unc_max']:.3e}" if np.isfinite(v["phi0_unc_max"]) else "—"
            tag = "  ← additive (ShapleyBO's target)" if nm == "CB" else ""
            print(f"    {nm:5s} {lu:>18s} {lv:>22s} {pm:>22s}{tag}")

    print("\n  Interpretation: `L_U = 0` must hold for **every** acquisition —")
    print("  A₀ = u(μ) and σ-only actions do not touch μ (regardless of distribution/shape).")
    print("  `L_V` is 0 only for additive (CB). A nonzero value under a nonlinear acquisition")
    print("  is not leakage but the **μ–σ interaction**: the value of uncertainty changes")
    print("  as one approaches the target. Capturing that interaction is the U channel's job.")

    # ── E0 design check ───────────────────────────────────────────────────
    bad, tot, worst = _constraint_check()
    print(f"\n═══ E0  action validity (design check) ═══")
    print(f"  {bad} constraint violations among {tot:,} coalitions "
          f"({100.0*bad/max(tot,1):.4f}%) · max sum error {worst:.2e}")

    # ── E1 + E2 real data ────────────────────────────────────────────────
    from tracebo.data import load as _load
    from tracebo.meta import SUITE_FULL
    from tracebo.targets import TARGET_QUANTILES
    dss = a.datasets.split(",") if a.datasets else list(SUITE_FULL)
    dss = [d for d in dss if _load(d).n >= a.n0 + a.budget + 20]
    jobs = [(ds, q, s, a.n0, a.budget)
            for ds, q, s in itertools.product(dss, TARGET_QUANTILES,
                                              range(a.seeds))]
    print(f"\n═══ E1·E2  real data ({len(jobs)} episodes) ═══", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for rs in ex.map(_episode, jobs, chunksize=2) for r in rs]
    print(f"  {len(rows)} iterations · {time.perf_counter()-t0:.0f}s\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"  E1  reconstruction residual |φ^A − φ^A₀ − φ^U| / max|φ^A|")
    print(f"        max {df.e1_rel.max():.3e} · median {df.e1_rel.median():.3e}")
    print(f"\n  E2-a  value-channel agreement")
    print(f"        EXP·OPT·TS  max|φ^A₀ − φ^A₀'|  max {df.dvalue_shared.max():.3e}")
    print(f"        t-EI        normalized deviation  mean {df.dvalue_tei.mean():.4f}"
          f" · max {df.dvalue_tei.max():.4f}")
    print(f"\n  E2-b  uncertainty channel — per acquisition pair")
    print(f"        {'pair':12s} {'Spearman ρ':>12s} {'sign disagr.':>12s}")
    for a1, a2 in PAIRS:
        r, s = df[f"rho_{a1}_{a2}"], df[f"sgn_{a1}_{a2}"]
        print(f"        {a1+'–'+a2:12s} {np.nanmean(r):12.4f} {np.nanmean(s):12.4f}")

    np.savez_compressed(a.out, cols=np.array(list(df.columns)),
                        rows=df.astype(str).to_numpy(),
                        synth=json.dumps({k: _synth_case(k)
                                          for k in ("split", "sigma")},
                                         default=float),
                        constraint=json.dumps(dict(bad=bad, total=tot,
                                                   worst=worst)))
    print(f"\n  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
