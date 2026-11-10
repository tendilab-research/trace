"""Hierarchical bootstrap — CI-A / CI-B (plan §6.3).

```
hierarchy: family(7) → plate → target(3) → seed,  10,000 replicates, paired
```

**CI-A** resamples families → "does it generalize to new experimental domains".
With only 7 families it is **necessarily wide, and we do not hide that**. The
"paired 95% CI excludes 0" criterion in spec §12 refers to this one.

**CI-B** holds families fixed and resamples only seeds → "the effect within these domains".

Why the plate level is required: `oer_plate_3496` and `4098` **share the same
X pool and differ only in the y surface**. If plates within a drawn family are
not resampled, the two plates are counted as independent samples and the
effective sample size of the OER family is inflated 2x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CELL = ["dataset", "family", "target_q", "surrogate", "acquisition", "seed"]


def _macro(d: pd.DataFrame) -> float:
    """Family-level macro-average (§6.2).

    Folds dataset→target→seed within each family. A plain mean would weight a
    family with 2 plates, like OER, twice.
    """
    per_ds = d.groupby(["family", "dataset", "target_q"])["delta"].mean()
    per_fam = per_ds.groupby(["family", "dataset"]).mean().groupby("family").mean()
    return float(per_fam.mean())


def hierarchical_ci(pairs: pd.DataFrame, n_boot: int = 10_000,
                    mode: str = "A", seed: int = 0,
                    alpha: float = 0.05) -> dict:
    """`pairs` is the output of `aggregate.paired()` (delta column required).

    mode="A" resamples from the family level; mode="B" holds families fixed and resamples only seeds.
    """
    rng = np.random.default_rng(seed)
    fams = pairs["family"].unique()
    by_fam = {f: g for f, g in pairs.groupby("family")}

    stats = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(fams, size=len(fams), replace=True) if mode == "A" else fams
        chunks = []
        for rep, f in enumerate(picked):
            g = by_fam[f]
            plates = g["dataset"].unique()
            # resample plates within the family — required because of the shared X pool
            pl = rng.choice(plates, size=len(plates), replace=True)
            for p in pl:
                gp = g[g["dataset"] == p]
                tq = gp["target_q"].unique()
                for q in rng.choice(tq, size=len(tq), replace=True):
                    gq = gp[gp["target_q"] == q]
                    take = rng.integers(0, len(gq), size=len(gq))
                    c = gq.iloc[take].copy()
                    # when resampling draws the same family multiple times, count
                    # each draw as a distinct unit so the macro-average is not distorted
                    c["family"] = f"{f}#{rep}"
                    chunks.append(c)
        stats[b] = _macro(pd.concat(chunks, ignore_index=True))

    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": _macro(pairs), "lo": float(lo), "hi": float(hi),
            "mode": mode, "n_boot": n_boot,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni. RQ2 uses m=12 (cells), RQ4 uses m=5 (controls) (§6.4)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k] = adj
        prev = adj
    return out
