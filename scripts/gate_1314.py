"""Gates 13·14 — approximation-approval gates that need a GPU (plan §5).

  Gate 13  TabPFN `n_estimators` 4 vs 8: **selected-candidate agreement ≥99%**
           → approve 4. Using 8 doubles the GPU-lane cost, so failing this
           gate changes the budget plan wholesale (§7.3).

  Gate 14  argmax of the top-K shortlist (K=256) **agrees ≥99% with the
           full-pool argmax**. Confirms empirically that the G15
           approximation (attribution on the top K only) does not change
           the argmax.

Both gates have the same shape: "do policy A and policy B pick the same
candidate". So each episode is run **once** (following one policy), and at
every iteration the other policy's choice is computed additionally and
compared. Running two trajectories separately would diverge the state and
the agreement rate would lose meaning — it must be a same-state comparison.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process, get_context

import numpy as np


def _pin_gpu(n_gpu: int) -> None:
    """Pin workers to GPUs round-robin.

    If everyone uses cuda:0, the SMs serialize and 6 cards become 1 even with
    80GB to spare. CUDA_VISIBLE_DEVICES must be set **before torch is
    imported**, hence spawn context + initializer (the top of this file does
    not import torch — that is the precondition).
    """
    i = (current_process()._identity or (1,))[0] - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i % max(n_gpu, 1))


def _agree(job):
    gate, ds, q, seed, surr, acq, n0, budget, K, policy = job
    from tracebo.acquisitions import ablation_seed, acquisitions, target_utility, crn
    from tracebo.data import load
    from tracebo.guards import OracleGuard
    from tracebo.policies import pick_reference, select_trace
    from tracebo.runner import Episode, _make_surrogate
    from tracebo.targets import initial_design, make_target

    pool = load(ds)
    target = make_target(pool, q)
    ep = Episode(ds, q, surr, acq, "coherence", n0, seed, budget, K)
    guard = OracleGuard(pool.y)
    guard.observe_many(initial_design(pool, target, n0, seed))
    X, ys = pool.X, target.y_star

    same = tot = 0
    for t in range(1, budget + 1):
        obs = guard.observed
        y_obs = guard.y_of(obs)
        e_best = float(np.abs(y_obs - ys).min())
        unobs = guard.unobserved(pool.n)
        if len(unobs) == 0:
            break
        rng = crn(seed, t, ep.crn_key)

        if gate == 13:
            sA = _make_surrogate(surr, seed).fit(X[obs], y_obs)          # n_est=4
            from tracebo.surrogates.tabpfn import TabPFNSurrogate
            sB = TabPFNSurrogate(n_estimators=8, seed=seed).fit(X[obs], y_obs)
            KA = KB = K
        else:                                                            # gate 14
            sA = sB = _make_surrogate(surr, seed).fit(X[obs], y_obs)
            KA, KB = K, len(unobs)                                       # top-K vs full

        ref = X[pick_reference(guard, ys)]
        asd = ablation_seed(ep.crn_key, t)
        picks = []
        for s_, K_ in ((sA, KA), (sB, KB)):
            r = crn(seed, t, ep.crn_key)     # **same** randomness for both policies (spirit of gate 10)
            a = acquisitions(s_, X[unobs], ys, e_best, r)[acq]
            if policy == "base":
                # Drop the re-ranking and compare acquisition argmax only. In
                # gate 13 this is the only way to tell whether disagreement is
                # due to the approximation (n_estimators) or the re-ranking machinery.
                picks.append(int(unobs[int(np.argmax(a))]))
                continue
            sel = select_trace(a, unobs, ref, X, pool.meta,
                               lambda Z: target_utility(s_, Z, ys, r),
                               K=K_, score="coherence", ablation_seed=asd)
            picks.append(sel.idx)

        same += int(picks[0] == picks[1]); tot += 1
        guard.observe(picks[0])              # follow policy A
        del rng
    return dict(gate=gate, dataset=ds, q=q, seed=seed, same=same, tot=tot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, required=True, choices=(13, 14))
    ap.add_argument("--datasets", default="oer_plate_3496")
    ap.add_argument("--surrogate", default=None)
    ap.add_argument("--acquisition", default="tEI")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--n0", type=int, default=5)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--n-gpu", type=int, default=0, help=">0 pins workers to GPUs")
    ap.add_argument("--policy", default="coherence", choices=("coherence", "base"),
                    help="base = compare acquisition argmax only, no re-ranking (isolate the cause)")
    a = ap.parse_args(argv)
    surr = a.surrogate or ("TabPFN" if a.gate == 13 else "GP")

    from tracebo.targets import TARGET_QUANTILES
    jobs = [(a.gate, ds, q, s, surr, a.acquisition, a.n0, a.budget, a.K, a.policy)
            for ds, q, s in itertools.product(
                a.datasets.split(","), TARGET_QUANTILES, range(a.seeds))]
    print(f"gate {a.gate} · {surr}/{a.acquisition} · policy={a.policy} · "
          f"episodes {len(jobs)} · workers {a.workers}", flush=True)

    t0 = time.perf_counter()
    kw = {}
    if a.n_gpu > 0:
        kw = dict(mp_context=get_context("spawn"),
                  initializer=_pin_gpu, initargs=(a.n_gpu,))
    with ProcessPoolExecutor(max_workers=a.workers, **kw) as ex:
        res = list(ex.map(_agree, jobs, chunksize=1))
    same = sum(r["same"] for r in res); tot = sum(r["tot"] for r in res)
    rate = same / max(tot, 1)
    print(f"done {time.perf_counter() - t0:.0f}s")

    label = ("n_estimators 4 vs 8" if a.gate == 13
             else f"K={a.K} vs full-pool")
    print("\n" + "=" * 66)
    print(f"  gate {a.gate} — {label}")
    print("=" * 66)
    for ds in a.datasets.split(","):
        for q in sorted({r["q"] for r in res}):
            rs = [r for r in res if r["dataset"] == ds and r["q"] == q]
            s_, t_ = sum(r["same"] for r in rs), sum(r["tot"] for r in rs)
            print(f"  {ds:<16} q={q:.1f}  {s_:>5}/{t_:<5} = {s_/max(t_,1):>7.4f}")
    ok = rate >= 0.99
    print(f"\n  overall agreement {same}/{tot} = {rate:.4f}  "
          f"→ {'PASS' if ok else 'FAIL'} (threshold ≥0.99)")
    if not ok:
        print("  ↳ The approximation changes the argmax. Running with it, one cannot")
        print("    claim 'the result is unchanged with less computation'.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
