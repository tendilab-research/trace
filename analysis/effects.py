"""Main-effect model and multiple comparisons (plan §6.4).

```python
smf.mixedlm("auc ~ correction*acquisition + correction*surrogate",
            data=df, groups=df["family_target"]).fit()
```

`groups` is `family:target` because observations cluster at that unit. Seeds
within the same (family, target) share **the same y surface and the same
target value**, so they are not independent. Ignoring this and using OLS
underestimates the standard errors.

Success@B is binomial and needs a separate path. The plan lists R
`lme4::glmer` first, with GEE + cluster-robust SE as the fallback. There is no
reason to add an R dependency today, so only GEE is implemented — the cluster
is likewise `family:target`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prep(df: pd.DataFrame, treat: str, control: str) -> pd.DataFrame:
    d = df[df["variant"].isin([treat, control])].copy()
    d["correction"] = (d["variant"] == treat).astype(int)
    d["family_target"] = (d["family"].astype(str) + ":"
                          + d["target_q"].astype(str))
    return d


def main_effect(df: pd.DataFrame, treat: str = "coherence",
                control: str = "base", metric: str = "auc_err_n"):
    """Mixed-effects model. Returns the statsmodels result object."""
    import statsmodels.formula.api as smf

    d = _prep(df, treat, control)
    d = d.rename(columns={metric: "y"})
    m = smf.mixedlm("y ~ correction * acquisition + correction * surrogate",
                    data=d, groups=d["family_target"])
    return m.fit(method="lbfgs")


def success_effect(df: pd.DataFrame, treat: str = "coherence",
                   control: str = "base"):
    """Success@B (binomial) — GEE + cluster-robust SE, cluster = family:target."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = _prep(df, treat, control)
    d["y"] = d["succ"].astype(int)
    m = smf.gee("y ~ correction * acquisition + correction * surrogate",
                groups="family_target", data=d,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable())
    return m.fit()


def per_cell(df: pd.DataFrame, treat: str = "coherence",
             control: str = "base", metric: str = "auc_err_n") -> pd.DataFrame:
    """Paired effect per 12 cells (surrogate × acquisition) + Holm(m=12) (§6.4).

    Within a cell, pair on (dataset, target, seed) and run Wilcoxon. Pairing is
    valid because D₀ and CRN are determined independently of the variant
    (gates 8·10).
    """
    from scipy.stats import wilcoxon

    from .bootstrap import holm

    keys = ["dataset", "family", "target_q", "seed"]
    rows, pv = [], {}
    for (s, a), g in df.groupby(["surrogate", "acquisition"]):
        A = g[g["variant"] == treat].set_index(keys)[metric]
        B = g[g["variant"] == control].set_index(keys)[metric]
        idx = A.index.intersection(B.index)
        if len(idx) < 6:
            continue
        d = (A.loc[idx] - B.loc[idx]).values
        nz = d[np.abs(d) > 1e-12]
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
        cell = f"{s}/{a}"
        pv[cell] = p
        rows.append({"cell": cell, "surrogate": s, "acquisition": a,
                     "n_pairs": len(d), "delta": float(d.mean()),
                     "win_rate": float(np.mean(d < 0)), "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        adj = holm({k: v for k, v in pv.items() if v == v})
        out["p_holm"] = out["cell"].map(adj)
        out = out.sort_values("delta")
    return out
