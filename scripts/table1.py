"""Table 1 — measure exactness and cost per dataset (plan §8 V5).

Two things go into one table.

  · **Identity error** max_x |Σ_k φ_k − (A(x) − A(x_ref))|. All four acquisitions.
    If this is nonzero, the claim "the explanation is an exact decomposition of
    the decision" degrades to an approximation.
  · **Cost** coalition rows per iteration. V4 decomposes only the single selected
    candidate, so it is near `2^m`, but the earlier design that reranked the top K
    had to decompose all K candidates (K=64). Both make one surrogate call per
    iteration — row count is the cost.

Without `--out`, prints to screen only.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

ACQ4 = ("EXP", "OPT", "tEI", "TS")
K_OLD = 64


def _one(ds, n0, iters, seed=0):
    from tracebo.acquisitions import acq_values
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.policies import pick_reference
    from tracebo.runner import _make_surrogate
    from tracebo.shapley import attribute_batch, coalition_design
    from tracebo.targets import initial_design, make_target

    pool = load(ds); target = make_target(pool, 0.5)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star

    err = {n: 0.0 for n in ACQ4}
    rows_v4, rows_old, ms, secs = [], [], [], []
    for t in range(1, iters + 1):
        obs = guard.observed; yo = guard.y_of(obs)
        eb = float(np.abs(yo - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) < K_OLD + 1:
            break
        surr = _make_surrogate("GP", seed).fit(X[obs], yo)
        key = f"{ds}|{t}"
        acq = acq_values(surr, X[unobs], ys, eb, "tEI", key)
        order = np.argsort(-acq)
        pick = int(unobs[order[0]])
        ref = X[pick_reference(guard, ys)]

        for n in ACQ4:
            fn = lambda Z, n=n: acq_values(surr, Z, ys, eb, n, key)
            t0 = time.perf_counter()
            att = attribute_batch(ref, X[pick][None, :], pool.meta, fn)
            if n == "tEI":
                secs.append(time.perf_counter() - t0)
                rows_v4.append(att.n_unique); ms.append(int(att.m[0]))
            rhs = float(fn(X[pick][None, :])[0] - fn(ref[None, :])[0])
            lhs = float(att.phi[0].sum())
            err[n] = max(err[n], abs(lhs - rhs) / max(1.0, abs(rhs)))

        # rows the earlier design (top-K reranking) required — same iteration, same reference
        top = X[unobs[order[:K_OLD]]]
        rows_old.append(len(coalition_design(ref, top, pool.meta)[0]))
        guard.observe(pick)

    if not rows_v4:
        return None
    return dict(ds=ds, m=float(np.mean(ms)), v4=float(np.mean(rows_v4)),
                old=float(np.mean(rows_old)),
                ms=1e3 * float(np.mean(secs)),
                **{f"e_{n}": err[n] for n in ACQ4})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="figures/table1.tex")
    a = ap.parse_args(argv)

    from tracebo.data import load
    from tracebo.meta import SUITE_FULL
    rows = []
    for ds in SUITE_FULL:
        if load(ds).n < a.n0 * 2 + K_OLD + 10:
            continue
        r = _one(ds, a.n0, a.iters)
        if r:
            rows.append(r); print("  " + str({k: (f"{v:.3g}" if isinstance(v, float) else v)
                                              for k, v in r.items()}), flush=True)
    if not rows:
        print("measurement failed", file=sys.stderr); return 1

    emax = max(max(r[f"e_{n}"] for n in ACQ4) for r in rows)
    tot_v4 = np.mean([r["v4"] for r in rows])
    tot_old = np.mean([r["old"] for r in rows])
    print(f"\n  overall max identity error {emax:.2e} · mean rows {tot_v4:.0f} vs "
          f"{tot_old:.0f} ({tot_old/tot_v4:.0f}x)")

    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{coalition rows / iter} & &"
         r" max identity", r"\cmidrule(lr){3-4}",
         r"dataset & $\bar m$ & ours & rerank ($K{=}64$) & speedup"
         r" & error", r"\midrule"]
    for r in rows:
        e = max(r[f"e_{n}"] for n in ACQ4)
        L.append(f"{r['ds'].replace('_', chr(92)+'_')} & {r['m']:.1f} & "
                 f"{r['v4']:.0f} & {r['old']:.0f} & "
                 f"{r['old']/r['v4']:.0f}$\\times$ & "
                 f"\\num{{{e:.1e}}} \\\\")
    L += [r"\midrule",
          f"mean & {np.mean([r['m'] for r in rows]):.1f} & {tot_v4:.0f} & "
          f"{tot_old:.0f} & {tot_old/tot_v4:.0f}$\\times$ & "
          f"\\num{{{emax:.1e}}} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"  → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
