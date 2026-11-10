"""Ranking-fidelity metrics — the `ranking` table (plan §6.1, Fig 4).

**Without this table there is no Fig 4 at all.** And it cannot be added
retroactively once a run starts — the trajectory keeps only the selected
candidate, and the full ranking is gone the moment the iteration ends.

One thing is measured: **which is closer to the oracle ranking, the acquisition
ranking (base) or the reranked one (corrected).** The oracle ranking uses
unobserved y, so the policy can never see it; it is computed only here — hence
this module must not be in gate 9's POLICY_MODULES.

`onestep_regret` is the easiest to interpret: "how much farther from the target
was the candidate chosen this iteration than the best one available". AUC is
the outcome of the whole trajectory; this is **single-step decision quality**,
which separates cause from effect.
"""

from __future__ import annotations

import numpy as np


def _ndcg(order: np.ndarray, gain: np.ndarray, k: int = 10) -> float:
    """nDCG over the top k. gain must be higher-is-better (= −error)."""
    k = min(k, len(order))
    if k == 0:
        return float("nan")
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = gain[order[:k]]
    ideal = np.sort(gain)[::-1][:k]
    denom = float((ideal * disc).sum())
    # gain is mapped to [0,1], so denom<=0 only when all candidates tie.
    return float((g * disc).sum() / denom) if abs(denom) > 1e-15 else float("nan")


def ranking_row(acq: np.ndarray, s: np.ndarray | None, y_true: np.ndarray,
                y_star: float, shortlist: np.ndarray | None,
                n_unobs: int) -> dict:
    """Ranking fidelity for one iteration.

    Parameters
    ----------
    acq        acquisition scores of the top-K shortlist candidates (higher is better)
    s          reranking score `q_A·q_X`. None for the base variant
    y_true     **true y of the same candidates** — oracle. Used only here
    y_star     target value
    shortlist  positions within the full unobserved array. Recorded for diagnostics only
    n_unobs    total number of unobserved candidates this iteration
    """
    from scipy.stats import kendalltau, spearmanr

    err = np.abs(np.asarray(y_true, dtype=float) - y_star)
    gain = -err
    # nDCG is ill-defined for negative gain. Map to [0,1] with a rank-preserving
    # linear transform (all ties -> all 1 -> NaN via the denom check).
    rng = gain.max() - gain.min()
    gain01 = (gain - gain.min()) / rng if rng > 1e-15 else np.ones_like(gain)

    out = {"shortlist_size": int(len(acq)),
           "is_full_pool": bool(len(acq) >= n_unobs),
           "n_unobs": int(n_unobs)}

    best_err = float(err.min())
    for tag, score in (("base", np.asarray(acq, dtype=float)), ("corrected", s)):
        if score is None:
            out.update({f"spearman_{tag}": float("nan"),
                        f"kendall_{tag}": float("nan"),
                        f"ndcg10_{tag}": float("nan"),
                        f"onestep_regret_{tag}": float("nan")})
            continue
        score = np.asarray(score, dtype=float)
        order = np.argsort(-score, kind="stable")
        # Correlation is undefined if score or gain is all ties. scipy warns
        # and returns nan, so filter beforehand and insert nan quietly.
        ok = len(score) >= 3 and score.std() > 1e-15 and gain.std() > 1e-15
        sp = spearmanr(score, gain).correlation if ok else float("nan")
        kt = kendalltau(score, gain).correlation if ok else float("nan")
        out.update({
            f"spearman_{tag}": float(sp),
            f"kendall_{tag}": float(kt),
            f"ndcg10_{tag}": _ndcg(order, gain01, 10),
            # Regret of this single step, relative to the best available in the shortlist.
            f"onestep_regret_{tag}": float(err[order[0]] - best_err),
        })
    return out
