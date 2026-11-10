"""Family-level statistics for the headline quantities of uncert_split.py.

The seed-level p-values in ``uncert_split.py`` are one-sample Wilcoxon tests
against 0 over all 480 episodes (8 datasets x 3 target quantiles x 20 seeds).
Episodes from the same dataset share the same response surface, so those
p-values treat correlated samples as independent. This script complements
them with statistics whose sampling unit is the materials *family*
(7 families: the two OER plates share one X pool and are pooled).

For every quantity reported through ``_stat`` in uncert_split.py (plus the
four channel correlations reported there without a p-value) we compute:

1. per-family mean / median and the sign of each family's mean relative to
   the null (0), with "k of 7 families agree with the pooled direction";
2. a family-blocked hierarchical bootstrap CI of the pooled (macro) mean,
   family -> plate -> target quantile -> seed, B=10000, seed=0 (same scheme
   as analysis/bootstrap.py, mode "A"); and a cluster-robust alternative
   that treats the 7 family means as the sample (t-interval, df=6);
3. an exact two-sided binomial sign test on the 7 family directions.

Usage:  PYTHONPATH=src .venv/bin/python scripts/family_stats.py
Writes  results/family_stats.{npz,md} and prints the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t as student_t, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tracebo.meta import DATASETS  # noqa: E402

IN = ROOT / "results/v5_480.npz"
OUT = ROOT / "results/family_stats"
B, SEED, NULL = 10_000, 0, 0.0

# quantities reported with a seed-level Wilcoxon p in uncert_split.py, in the
# order they are printed there; the first four are printed there as means only
QUANT = ["r_u_sd", "r_u_mu", "r_v_sd", "r_v_mu",
         "unc_share", "ushare_EXP", "ushare_OPT", "ushare_tEI", "ushare_TS",
         "r_u_exp_opt", "r_u_exp_ts", "r_u_exp_tei",
         "tei_split", "tei_dev", "clip", "top1",
         "u_exp_x", "u_opt_x", "far", "relief_exp"]


def _nanmean(a, axis=0):
    return np.nanmean(a, axis=axis) if np.isfinite(a).any() else np.full(a.shape[1], np.nan)


def macro(cells, rng=None):
    """Macro-average over family -> plate -> target quantile -> seed.

    ``cells`` is {family: {plate: {q: (n_seed, K) array}}}. With ``rng`` every
    level is resampled with replacement (hierarchical bootstrap, mode A).
    """
    fams = list(cells)
    if rng is not None:
        fams = list(rng.choice(fams, size=len(fams), replace=True))
    fam_means = []
    for f in fams:
        plates = list(cells[f])
        if rng is not None:
            plates = list(rng.choice(plates, size=len(plates), replace=True))
        pl_means = []
        for p in plates:
            qs = list(cells[f][p])
            if rng is not None:
                qs = list(rng.choice(qs, size=len(qs), replace=True))
            q_means = []
            for q in qs:
                a = cells[f][p][q]
                if rng is not None:
                    a = a[rng.integers(0, len(a), size=len(a))]
                q_means.append(np.nanmean(a, axis=0))
            pl_means.append(np.mean(q_means, axis=0))
        fam_means.append(np.mean(pl_means, axis=0))
    return np.array(fam_means)  # (n_family, K)


def main() -> int:
    d = np.load(IN)
    cols = list(d["ep_cols"])
    ep, ds, qq = d["ep"], d["ep_dataset"], d["ep_q"]
    ki = [cols.index(k) for k in QUANT]
    X = ep[:, ki]
    fam_of = {k: v.family for k, v in DATASETS.items()}
    families = sorted({fam_of[s] for s in ds})
    cells = {f: {} for f in families}
    for p in sorted(set(ds)):
        for q in sorted(set(qq)):
            m = (ds == p) & (qq == q)
            cells[fam_of[p]].setdefault(p, {})[float(q)] = X[m]

    # 1. pooled (episode-level, as in uncert_split.py) + seed-level Wilcoxon p
    pooled = np.nanmean(X, axis=0)
    p_seed = np.full(len(QUANT), np.nan)
    for j in range(len(QUANT)):
        v = X[:, j][np.isfinite(X[:, j])]
        try:
            p_seed[j] = wilcoxon(v - NULL).pvalue
        except ValueError:
            pass

    # per-family mean / median (plates folded first so OER is not double-counted)
    fam_mean = macro(cells)
    fam_med = np.array([np.nanmedian(np.concatenate(
        [a for p in cells[f].values() for a in p.values()]), axis=0)
        for f in families])
    pooled_sign = np.sign(pooled - NULL)
    fam_sign = np.sign(fam_mean - NULL)
    agree = (fam_sign == pooled_sign).sum(axis=0)
    n_fam = len(families)

    # 2a. family-blocked hierarchical bootstrap of the macro mean
    rng = np.random.default_rng(SEED)
    boot = np.array([macro(cells, rng).mean(axis=0) for _ in range(B)])
    bci = np.quantile(boot, [0.025, 0.975], axis=0)
    macro_mean = fam_mean.mean(axis=0)

    # 2b. cluster-robust: 7 family means as the sample, t-interval (df=6)
    se = fam_mean.std(axis=0, ddof=1) / np.sqrt(n_fam)
    tq = student_t.ppf(0.975, n_fam - 1)
    tci = np.array([macro_mean - tq * se, macro_mean + tq * se])

    # 3. exact sign test on the family directions
    p_fam = np.array([binomtest(int(k), n_fam, 0.5).pvalue for k in agree])

    def dirstr(j):
        return "".join("+" if s > 0 else "-" if s < 0 else "0" for s in fam_sign[:, j])

    hdr = ["quantity", "pooled mean", "seed-level p", "family direction",
           "k/7", "blocked-bootstrap 95% CI", "family t-CI (df=6)",
           "family sign-test p"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for j, k in enumerate(QUANT):
        lines.append(
            f"| {k} | {pooled[j]:+.4f} | {p_seed[j]:.2g} | `{dirstr(j)}` | "
            f"{agree[j]}/{n_fam} | [{bci[0, j]:+.4f}, {bci[1, j]:+.4f}] | "
            f"[{tci[0, j]:+.4f}, {tci[1, j]:+.4f}] | {p_fam[j]:.3f} |")
    md = "\n".join([
        "# Family-level statistics (uncert_split headline quantities)", "",
        f"Families ({n_fam}, order of the direction string): "
        + ", ".join(families) + ".",
        f"Null = {NULL:g} for every row (as in uncert_split.py's one-sample "
        "Wilcoxon). Pooled mean = episode mean over all 480 episodes; the two "
        "CIs are for the family macro-average (OER plates folded first). "
        f"Blocked bootstrap: family -> plate -> target -> seed, B={B}, seed={SEED}. "
        "Family t-CI: 7 family means, t(0.975, 6). Sign test: exact two-sided "
        "binomial on k/7.", "", *lines, ""])
    print(md)

    table = np.column_stack([pooled, p_seed, agree, bci[0], bci[1],
                             tci[0], tci[1], p_fam, macro_mean])
    np.savez_compressed(
        OUT.with_suffix(".npz"), rows=np.array(QUANT), table=table,
        cols=np.array(["pooled_mean", "p_seed", "k_agree", "boot_lo", "boot_hi",
                       "t_lo", "t_hi", "p_family_sign", "macro_mean"]),
        families=np.array(families), fam_mean=fam_mean, fam_median=fam_med,
        fam_sign=fam_sign, null=NULL, B=B, seed=SEED)
    OUT.with_suffix(".md").write_text(md)
    print(f"-> {OUT}.npz / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
