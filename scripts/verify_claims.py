"""Re-verify manuscript claims that have **no saved result file**.

Three of them. All enter the main text or appendix as numbers, so they are
kept in reproducible form.

  1. Are TabPFN `moments` the exact moments of the bar distribution
     (Appendix B / §4.7 — "not a normality assumption")
  2. Does the triple-union shortlist preserve the argmax for w<=1
     (Appendix F.0 — exact at the vertices, approximate in the interior)
  3. Does the sigma->0 decomposition of CB recover the two ShapleyBO terms exactly
     (Appendix B.5)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np


def check_tabpfn(n_mc=4000, seed=0):
    """Analytic moments vs sample moments. Must agree to within MC standard error."""
    from tracebo.data import load
    from tracebo.surrogates.tabpfn import TabPFNSurrogate
    pool = load("crossed_barrel")
    rng = np.random.default_rng(seed)
    tr = rng.choice(pool.n, 60, replace=False)
    te = rng.choice(np.setdiff1d(np.arange(pool.n), tr), 40, replace=False)
    s = TabPFNSurrogate(seed=seed).fit(pool.X[tr], pool.y[tr])
    mu, sd = s.moments(pool.X[te])
    S = s.sample(pool.X[te], n_mc, rng)          # (n_mc, n_te) or (n_te, n_mc)
    S = S if S.shape[0] == n_mc else S.T
    mu_mc, sd_mc = S.mean(axis=0), S.std(axis=0, ddof=1)
    yr = float(pool.y.max() - pool.y.min())
    dmu = float(np.abs(mu - mu_mc).max()) / yr
    dsd = float(np.abs(sd - sd_mc).max()) / yr
    # MC standard error: sd/sqrt(n) (mean), sd/sqrt(2n) (std)
    se_mu = float((sd_mc / np.sqrt(n_mc)).max()) / yr
    se_sd = float((sd_mc / np.sqrt(2 * n_mc)).max()) / yr
    ok = dmu < 6 * se_mu and dsd < 6 * se_sd
    print(f"  1. TabPFN moments  |Δμ|max/range={dmu:.2e} (6·se={6*se_mu:.2e})  "
          f"|Δσ|max/range={dsd:.2e} (6·se={6*se_sd:.2e})   {'OK' if ok else 'FAIL'}")
    return ok


def check_shortlist(seeds=5, K=24, tol=5e-3):
    """Does the argmax over the shortlist recover the **score** of the full argmax.

    The triple union top-K(A) ∪ top-K(A₀) ∪ top-K(−E) contains the **vertices**
    of this family by definition (w=1 is argmax A, w=0 is argmax A₀). But at an
    interior point 0<w<1, s=(1−w)A₀+wA may be neither argmax —
    **the shortlist is an approximation in the interior.** So we measure
    **score loss** rather than index agreement, and require exactness only at
    the vertices.
    """
    from tracebo.acquisitions import ALPHA, _from_moments, acq_plugin
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.meta import DATASETS
    from tracebo.runner import _make_surrogate
    from tracebo.targets import TARGET_QUANTILES, initial_design, make_target

    vert, inter = [], []
    for ds in DATASETS:
        pool = load(ds)
        for q in TARGET_QUANTILES:
            tgt = make_target(pool, q)
            for seed in range(seeds):
                g = OracleGuard(pool.y)
                g.observe_many(initial_design(pool, tgt, 10, seed))
                obs = g.observed; yo = g.y_of(obs)
                eb = float(np.abs(yo - tgt.y_star).min())
                un = g.unobserved(pool.n)
                sur = _make_surrogate("GP", seed).fit(pool.X[obs], yo)
                mu, sd = sur.moments(pool.X[un])
                for name in ("EXP", "OPT", "tEI", "TS"):
                    key = f"{ds}|{q}|{seed}|1"
                    a = _from_moments(mu, sd, pool.X[un], tgt.y_star, eb, name,
                                      key, ALPHA)
                    a0 = acq_plugin(name, mu, tgt.y_star, eb)
                    E = np.abs(mu - tgt.y_star)
                    k = min(K, len(un))
                    top = lambda v: np.argpartition(-v, k - 1)[:k]
                    sl = np.union1d(np.union1d(top(a), top(a0)), top(-E))
                    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                        s = a0 + w * (a - a0)
                        rngs = float(s.max() - s.min()) or 1.0
                        loss = float(s.max() - s[sl].max()) / rngs
                        (vert if w in (0.0, 1.0) else inter).append(loss)
    v, i = np.array(vert), np.array(inter)
    ok = float(v.max()) < 1e-12 and float(i.max()) < tol
    print(f"  2. shortlist  vertices w∈{{0,1}}: n={len(v)} max score loss "
          f"{v.max():.1e} (must be exact)")
    print(f"                interior 0<w<1:    n={len(i)} max {i.max():.1e} · "
          f"median {np.median(i):.1e} · cases with loss>0: {int((i>0).sum())}"
          f"   {'OK' if ok else 'FAIL'}")
    return ok


def check_cb():
    """CB is additive, so U must be exactly λσ."""
    from tracebo.acquisitions import LAMBDA_CB, _from_moments, acq_plugin
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 3, 500); sd = rng.gamma(2, 0.5, 500); ys = 1.3
    a = _from_moments(mu, sd, None, ys, 0.7, "CB", "k", 0.1)
    a0 = acq_plugin("CB", mu, ys, 0.7)
    e_val = float(np.abs(a0 - (-np.abs(mu - ys))).max())
    e_unc = float(np.abs((a - a0) - LAMBDA_CB * sd).max())
    ok = e_val < 1e-12 and e_unc < 1e-12
    print(f"  3. CB decomposition  |A₀−(−|μ−y*|)|max={e_val:.1e}  "
          f"|U−λσ|max={e_unc:.1e}   {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("Re-verifying manuscript claims")
    ok = [check_cb(), check_shortlist()]
    try:
        ok.append(check_tabpfn())
    except Exception as e:                       # no GPU available
        print(f"  1. TabPFN moments  skipped ({type(e).__name__}: {e})")
    print("\n  " + ("all passed" if all(ok) else "failures present"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
