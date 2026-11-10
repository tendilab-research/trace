"""V5 — split the acquisition into a **value channel / uncertainty channel** (plan §8).

    A₀(x) = lim_{σ→0} A(x)        U(x) = A(x) − A₀(x)
    φ_k^A = φ_k^{A₀} + φ_k^U      ← Shapley linearity, not an approximation (gate 18)

Standard BO only says "this candidate was chosen because it is uncertain". It
cannot say **which change is the uncertain part**. That cannot come from
decomposing the utility Û (there is no exploration term to begin with); only
decomposing the acquisition yields it.

An earlier attempt used `OPT − EXP`. That is the wrong pair — `EXP = E[−|Y−y*|]`
is a nonlinear expectation and **already absorbs σ**. The true deterministic
counterpart is the σ→0 limit, and with that choice EXP·OPT·TS share **the same
A₀ = −|μ−y*|** (tEI is a constant shift when the clip is inactive, so φ is the
same, gate 22). Hence

    φ^A − φ^{A'} = φ^{U_A} − φ^{U_{A'}}      ← identity

and the statement "the choice of acquisition is entirely a choice of how to
price uncertainty" can be made at the level of individual actions.

What is measured:

  [R2] Discriminant validity — 2×2. Predictions:
         corr(φ^U , Δσ)   high      corr(φ^{A₀}, Δσ)   ≈ 0
         corr(φ^{A₀}, ψ₀) high      corr(φ^U , ψ₀)     ≈ 0
       Only the diagonal holding means the two channels really measure different things.
       Δσ_k and ψ₀_k are both already present in **the singleton rows of the same coalition matrix**.
  [R3] Dependence of the uncertainty share Σ|φ^U| / (Σ|φ^{A₀}| + Σ|φ^U|) on iteration t.
  [R4] Do the four acquisitions' φ^U differ (φ^{A₀} equality is confirmed empirically).
  [R5] top-1 |φ^A| share — is the explanation actionable.
  [sign] In target matching, uncertainty cuts both ways. U_EXP ≤ 0 always,
         U_OPT is positive only when |μ−y*|/σ > t* (gate 21).

**Cost 0.** A single `moments(Z)` call yields all eight channels of the four acquisitions.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ACQ4 = ("EXP", "OPT", "tEI", "TS")
DRIVER = "tEI"          # acquisition that drives the trajectory (selection is pure argmax = V4)


def _corr(a, b, kind="spearman"):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).correlation if kind == "spearman"
                 else pearsonr(a, b)[0])


def _pcorr(y, x, z):
    """Rank partial correlation of y·x controlling for z. Rank-transform, then linear residualization."""
    from scipy.stats import rankdata
    y, x, z = (np.asarray(a, float) for a in (y, x, z))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 5:
        return float("nan")
    Y, Xv, Zv = (rankdata(a[ok]).astype(float) for a in (y, x, z))
    if Zv.std() < 1e-12:
        return _corr(Y, Xv, "pearson")
    res = lambda v: v - np.polyval(np.polyfit(Zv, v, 1), Zv)
    return _corr(res(Y), res(Xv), "pearson")


def _one(job):
    ds, q, seed, surr_name, n0, budget = job
    from tracebo.acquisitions import (ALPHA, _from_moments, acq_plugin,
                                      acq_values, opt_sign_threshold,
                                      tei_clip_active)
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
    t_star = opt_sign_threshold(ALPHA)

    rec, ep, case = [], [], None
    for t in range(1, budget + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < 2:
            break
        surr = _make_surrogate(surr_name, seed).fit(X[obs], yo)
        key = f"{ds}|{q}|{seed}|{t}"
        acq = acq_values(surr, X[unobs], ys, eb, DRIVER, key)
        pick = int(unobs[int(np.argmax(acq))])
        ref = X[pick_reference(guard, ys)]

        # ── one coalition matrix, one moments call → four acquisitions × three channels
        Z, rows, acts = coalition_design(ref, X[pick][None, :], pool.meta)
        mom = surr.moments(Z)
        if mom is None:
            raise NotImplementedError(f"{surr_name}: moments required")
        mu, sd = mom
        r, a0s = rows[0], acts[0]
        m = len(a0s)
        if m < 2:
            guard.observe(pick); continue
        sing = r[1 << np.arange(m)]                  # singleton row indices

        # ── right-hand side: counterfactual singleton effects moving **μ only / σ only**.
        # Initially Δσ = σ(ref+{k}) − σ(ref) was used as the RHS of the uncertainty channel.
        # That is a contaminated instrument — moving away from the reference moves σ and
        # |μ−y*| together, so even A₀, **a function of μ alone**, correlates +0.30 with Δσ.
        # So we use counterfactuals that swap only one side. No extra surrogate calls.
        mu0, sd0 = float(mu[r[0]]), float(sd[r[0]])
        Zs = Z[sing]
        o = np.ones(m)
        a00 = _from_moments(mu0 * o, sd0 * o, Zs, ys, eb, DRIVER, key, ALPHA)
        psi_mu = _from_moments(mu[sing], sd0 * o, Zs, ys, eb,
                               DRIVER, key, ALPHA) - a00      # value RHS
        psi_sd = _from_moments(mu0 * o, sd[sing], Zs, ys, eb,
                               DRIVER, key, ALPHA) - a00      # uncertainty RHS
        dsd = sd[sing] - sd0                         # contaminated instrument — for contrast

        phi0_ref, chan = None, {}
        for name in ACQ4:
            a = _from_moments(mu, sd, Z, ys, eb, name, key, ALPHA)
            a0 = acq_plugin(name, mu, ys, eb)
            pa = exact_shapley(a[r], m)
            p0 = exact_shapley(a0[r], m)
            pu = pa - p0
            chan[name] = (pa, p0, pu)
            if name != "tEI":
                phi0_ref = p0 if phi0_ref is None else phi0_ref

        # φ^{A₀} sharing — holds **only for EXP·OPT·TS**. The three are the mean/quantile/sample
        # of the same utility u = −|y−y*|, i.e. location functionals, and are indistinguishable
        # under a degenerate posterior. tEI adds a clip on top of u, so
        # A₀ = max(0, e+u(μ)) and it becomes **a different function** the moment the clip
        # engages (same structure as EI₀ ≠ UCB₀). That deviation is measured here.
        share_err = max(float(np.abs(chan[n][1] - phi0_ref).max())
                        for n in ("EXP", "OPT", "TS"))
        nrm = lambda v: v / max(float(np.abs(v).sum()), 1e-30)
        tei_dev = float(np.abs(nrm(chan["tEI"][1]) - nrm(phi0_ref)).max())
        clip_c = tei_clip_active(mu[r], ys, eb)
        pa_d, p0_d, pu_d = chan[DRIVER]
        ratio = np.abs(mu[r[0]] - ys) / sd[r[0]]     # |μ−y*|/σ of the reference recipe

        for k in range(m):
            rec.append((float(p0_d[k]), float(pu_d[k]), float(psi_sd[k]),
                        float(psi_mu[k]), float(dsd[k]), a0s[k].kind,
                        *[float(chan[n][2][k]) for n in ACQ4]))
        # Case study (Fig 1). Do **not** pick the iteration with the most actions —
        # an m=9 case has unreadable labels, and the two channels happen to differ
        # 100x in magnitude so it looks like "the shared channel is 0". Require both
        # a readable m and **comparable channel magnitudes**.
        s0 = float(np.abs(p0_d).sum()); sU = float(np.abs(pu_d).sum())
        bal = min(s0, sU) / max(s0, sU, 1e-30)
        score = bal if 4 <= m <= 6 else -1.0
        if case is None or score > case["score"]:
            case = dict(score=score, bal=bal,dataset=ds, q=q, seed=seed, t=t, m=m,
                        labels=[a.label(pool.meta) for a in a0s],
                        kinds=[a.kind for a in a0s],
                        phi0=p0_d.tolist(),
                        phi0all={n: chan[n][1].tolist() for n in ACQ4},
                        clip=clip_c, tei_dev=tei_dev,
                        phiU={n: chan[n][2].tolist() for n in ACQ4},
                        phiA={n: chan[n][0].tolist() for n in ACQ4},
                        y_star=float(ys), e_best=float(eb),
                        mu_ref=float(mu[r[0]]), sd_ref=float(sd[r[0]]),
                        mu_cand=float(mu[r[-1]]), sd_cand=float(sd[r[-1]]))
        ep.append(dict(
            t=t, m=m,
            # [R2] channel descriptives — **not a validity test** (see note in the paper).
            # On real actions ψ^σ and ψ^μ are correlated (a change moving away from the
            # data moves σ and μ together), so marginal correlations have no
            # discriminative power. The size of that confound is reported alongside.
            confound=_corr(psi_sd, psi_mu),
            r_u_sd=_corr(pu_d, psi_sd),    r_v_sd=_corr(p0_d, psi_sd),
            r_v_mu=_corr(p0_d, psi_mu),    r_u_mu=_corr(pu_d, psi_mu),
            p_u_sd=_pcorr(pu_d, psi_sd, psi_mu),   # controlling ψ^μ
            p_v_mu=_pcorr(p0_d, psi_mu, psi_sd),   # controlling ψ^σ
            # [R3] uncertainty share
            unc_share=float(np.abs(pu_d).sum()
                            / max(np.abs(pu_d).sum() + np.abs(p0_d).sum(),
                                  1e-30)),
            # Channel share per acquisition. EXP·OPT·TS use the **same** φ^{A₀}
            # (Prop. shared), so the second term of the denominator is equal and the
            # difference comes entirely from Σ|φ^U|. Only t-EI uses its own φ^{A₀}.
            **{f"ushare_{n}": float(
                np.abs(chan[n][2]).sum()
                / max(np.abs(chan[n][2]).sum() + np.abs(chan[n][1]).sum(), 1e-30))
               for n in ACQ4},
            ratio=float(ratio),
            # [R4] do the four U channels differ
            r_u_exp_opt=_corr(chan["EXP"][2], chan["OPT"][2]),
            r_u_exp_ts=_corr(chan["EXP"][2], chan["TS"][2]),
            r_u_exp_tei=_corr(chan["EXP"][2], chan["tEI"][2]),
            share_err=share_err,
            tei_dev=tei_dev,                   # tEI deviation from A₀ (normalized)
            tei_split=float(tei_dev > 1e-9),   # fraction of iterations that split
            clip=clip_c,
            # [R5]
            top1=float(np.abs(pa_d).max() / max(np.abs(pa_d).sum(), 1e-30)),
            eff_err=float(abs(pa_d.sum() - (
                _from_moments(mu, sd, Z, ys, eb, DRIVER, key, ALPHA)[r[-1]]
                - _from_moments(mu, sd, Z, ys, eb, DRIVER, key, ALPHA)[r[0]]))),
            # Sign — the sign of **U(x) itself**. φ^U_k decomposes U(x)−U(ref), so
            # mixed signs are expected there and it must not be used here (gate 21).
            u_exp_x=float(_from_moments(mu, sd, Z, ys, eb, "EXP", key, ALPHA)[r[-1]]
                          - acq_plugin("EXP", mu, ys, eb)[r[-1]]),
            u_opt_x=float(_from_moments(mu, sd, Z, ys, eb, "OPT", key, ALPHA)[r[-1]]
                          - acq_plugin("OPT", mu, ys, eb)[r[-1]]),
            # Relief contribution — φ^U_k > 0 means "this change makes the candidate
            # less risky than the reference". Interpretable on its own.
            relief_exp=float(np.mean(chan["EXP"][2] > 0)),
            far=float(ratio > t_star),
        ))
        guard.observe(pick)

    if not ep:
        return None
    keys = [k for k in ep[0] if k != "t"]
    A = np.array([[e[k] for k in keys] for e in ep], dtype=float)
    with np.errstate(invalid="ignore"):
        out = {k: (float(np.nanmean(A[:, i]))
                   if np.isfinite(A[:, i]).any() else float("nan"))
               for i, k in enumerate(keys)}
    out.update(dataset=ds, q=q, seed=seed, n_iter=len(ep), rec=rec, case=case,
               share_err=float(np.nanmax(A[:, keys.index("share_err")])),
               eff_err=float(np.nanmax(A[:, keys.index("eff_err")])),
               traj=[(e["t"], e["unc_share"], e["ratio"]) for e in ep])
    return out


def _stat(res, key, name, indent="  "):
    from scipy.stats import wilcoxon
    v = np.array([r[key] for r in res], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        print(f"{indent}{name:26s} insufficient samples ({len(v)})"); return
    try:
        p = wilcoxon(v).pvalue
    except ValueError:
        p = float("nan")
    print(f"{indent}{name:26s} mean {v.mean():+.4f}  median {np.median(v):+.4f}"
          f"  episodes {len(v)}  p={p:.3g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="GP")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--out", default=None, help="save results as .npz (for figures)")
    a = ap.parse_args(argv)

    from tracebo.data import load as _load
    from tracebo.meta import SUITE_FULL
    from tracebo.targets import TARGET_QUANTILES
    datasets = a.datasets.split(",") if a.datasets else list(SUITE_FULL)
    datasets = [d for d in datasets if _load(d).n >= a.n0 * 2 + 50]

    jobs = [(ds, q, s, a.surrogate, a.n0, a.budget)
            for ds, q, s in itertools.product(datasets, TARGET_QUANTILES,
                                              range(a.seeds))]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(_one, jobs, chunksize=2) if r]
    print(f"\n  {a.surrogate} · driven by {DRIVER} · n0={a.n0} · "
          f"episodes {len(res)} · {time.perf_counter()-t0:.0f}s", flush=True)

    rec = np.array([r[:5] + r[6:] for x in res for r in x["rec"]], dtype=float)
    kinds = [r[5] for x in res for r in x["rec"]]
    V, U, PSD, PMU, DSD = (rec[:, 0], rec[:, 1], rec[:, 2],
                           rec[:, 3], rec[:, 4])
    print(f"  action records n={len(rec)} (comp {kinds.count('comp')} · "
          f"proc {kinds.count('proc')})")
    print(f"  gate 16 max error {max(r['eff_err'] for r in res):.2e} · "
          f"gate 22 max error {max(r['share_err'] for r in res):.2e} · "
          f"tEI clip active {np.mean([r['clip'] for r in res]):.2f}")

    print("\n  [R2] channel descriptives — not a validity 'test'")
    print("    A₀ does not see σ, so by definition **all** of the acquisition's")
    print("    σ-dependence lives in U. An identity with nothing to test. On real")
    print("    actions ψ^σ and ψ^μ are confounded, so marginal correlations have no discriminative power.")
    print(f"    confound size corr(ψ^σ, ψ^μ) = {_corr(PSD, PMU):+.4f}  "
          f"(episode mean {np.nanmean([r['confound'] for r in res]):+.4f})")
    print(f"    {'':16s}{'ψ^σ (σ only)':>20s}{'ψ^μ (μ only)':>20s}")
    for lab, k1, k2 in (("φ^U   (uncert.)", "r_u_sd", "r_u_mu"),
                        ("φ^A₀  (value)", "r_v_sd", "r_v_mu")):
        m1 = np.nanmean([r[k1] for r in res])
        m2 = np.nanmean([r[k2] for r in res])
        print(f"    {lab:16s}{m1:>+20.4f}{m2:>+20.4f}")
    print(f"    pooled spearman:  φ^U·ψ^σ {_corr(U, PSD):+.4f}   "
          f"φ^A₀·ψ^σ {_corr(V, PSD):+.4f}   "
          f"φ^U·ψ^μ {_corr(U, PMU):+.4f}   φ^A₀·ψ^μ {_corr(V, PMU):+.4f}")
    # Signed correlation misses the confound — ψ^σ is effectively a magnitude variable,
    # so compare against |ψ^μ|. With 3–4 actions per episode, partial correlation is
    # meaningful only in the pool.
    print(f"    magnitude confound corr(ψ^σ, |ψ^μ|) = {_corr(PSD, np.abs(PMU)):+.4f}"
          f"   corr(φ^A₀, |ψ^μ|) = {_corr(V, np.abs(PMU)):+.4f}")
    print("    partial correlations (pool, controlling the other channel):")
    print(f"      pool: φ^U·ψ^σ|ψ^μ {_pcorr(U, PSD, PMU):+.4f}   "
          f"φ^A₀·ψ^μ|ψ^σ {_pcorr(V, PMU, PSD):+.4f}   "
          f"φ^A₀·ψ^σ|ψ^μ {_pcorr(V, PSD, PMU):+.4f}")
    print("    reference — contaminated instrument Δσ:  "
          f"φ^U·Δσ {_corr(U, DSD):+.4f}  φ^A₀·Δσ {_corr(V, DSD):+.4f}  "
          f"ψ^μ·Δσ {_corr(PMU, DSD):+.4f}")

    print("\n  [R3] share of the acquisition taken by uncertainty")
    _stat(res, "unc_share", "Σ|φ^U|/(Σ|φ^U|+Σ|φ^A₀|)")
    for _n in ACQ4:
        _stat(res, f"ushare_{_n}", f"channel share — {_n}")
    traj = {}
    for x in res:
        for t, s, _ in x["traj"]:
            traj.setdefault(t, []).append(s)
    line = "  ".join(f"t{t}:{np.mean(v):.3f}" for t, v in sorted(traj.items()))
    print(f"    per iteration — {line}")

    print("\n  [R4] ★ φ^{A₀} is **shared**; all differences between acquisitions are in the U channel")
    print("    φ^A − φ^A' = φ^U − φ^U' is an identity; the lower the correlations below,")
    print("    the more 'which acquisition is used' changes the explanation wholesale.")
    _stat(res, "r_u_exp_opt", "corr(U_EXP, U_OPT)")
    _stat(res, "r_u_exp_ts", "corr(U_EXP, U_TS)")
    _stat(res, "r_u_exp_tei", "corr(U_EXP, U_tEI)")

    print("\n  [R4b] how far does sharing go — tEI splits because of the clip")
    _stat(res, "tei_split", "fraction φ^A₀_tEI ≠ shared")
    _stat(res, "tei_dev", "deviation size (normalized max)")
    _stat(res, "clip", "tEI clip active rate")

    print("\n  [R5] is the explanation concentrated")
    _stat(res, "top1", "top1 |φ^A| share")

    print("\n  [sign] in target matching, uncertainty cuts both ways")
    _stat(res, "u_exp_x", "U_EXP(x)  (must be ≤0)")
    _stat(res, "u_opt_x", "U_OPT(x)  (sign varies)")
    _stat(res, "far", "fraction |μ−y*|/σ > t*")
    _stat(res, "relief_exp", "fraction of actions with φ^U_EXP > 0")

    if a.out:
        import json
        scalars = [k for k in res[0] if isinstance(res[0][k], float)]
        # Save the per-episode cases **all of them**. Which case plots well is only
        # known after plotting, and there is no reason to rerun a 3-minute sweep each time.
        cases = [x["case"] for x in res if x["case"]]
        ncase = max(cases, key=lambda c: c["score"])
        np.savez_compressed(
            a.out,
            action=rec, action_kind=np.array(kinds),
            action_cols=np.array(["phi0", "phiU", "psi_sd", "psi_mu", "dsd",
                                  *[f"phiU_{n}" for n in ACQ4]]),
            ep=np.array([[x[k] for k in scalars] for x in res]),
            ep_cols=np.array(scalars),
            ep_dataset=np.array([x["dataset"] for x in res]),
            ep_q=np.array([x["q"] for x in res]),
            traj=np.array([[t, u, r] for x in res for t, u, r in x["traj"]]),
            case=np.array([json.dumps(ncase)]),
            cases=np.array([json.dumps(c) for c in cases]))
        print(f"\n  → {a.out}  (actions {len(rec)} · episodes {len(res)} · "
              f"case {ncase['dataset']} m={ncase['m']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
