"""trajectory parquet → per-episode metrics (plan §6.1–6.2).

**`ε` and `B` are not stored** (§2.2). The policy depends on neither, so
recomputing them at aggregation time gives sensitivity analysis for free,
without re-running. Calling with a different `budget` argument is exactly
App A.

Normalization lives here because of a measurement. Naively averaging the raw
Target-error AUC over 8 datasets lets the single hplc (y range 0–2372) swallow
the seven OER (0.34–0.75) datasets — the measured |Δ| ratio was 50,000x. Only
after making it dimensionless by the evaluation-space y range does the
macro-average mean anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "family", "target_q", "surrogate", "acquisition",
        "variant", "seed"]


def _y_range(dataset: str) -> float:
    """y range in evaluation space. A pool constant, independent of the policy (not an oracle leak)."""
    from tracebo.data import load
    p = load(dataset)
    r = float(p.y.max() - p.y.min())
    return r if r > 0 else 1.0


def load_trajectory(root: Path, budget: int | None = None) -> pd.DataFrame:
    """Read all shard parquets and fold them to **one row per episode**."""
    import duckdb

    files = str(Path(root) / "trajectory" / "*.parquet")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{files}')").df()
    con.close()
    if budget is not None:
        df = df[df["t"] <= budget]

    df = df.sort_values(KEYS + ["t"])
    g = df.groupby(KEYS, sort=False)

    out = g.agg(
        n_iter=("t", "size"),
        succ=("is_hit", "any"),
        eval_space=("eval_space", "first"),
        tier=("tier", "first"),
        mean_m=("n_actions_m", "mean"),
        mean_rows=("n_surrogate_rows", "mean"),
        wall_s=("iter_wall_s", "sum"),
        corr_coh_du=("corr_coherence_vs_delta_util", "mean"),
    ).reset_index()

    # best-so-far curve is a cumulative min, so build it directly inside the groupby.
    bsf = df.groupby(KEYS, sort=False)["target_err"].cummin()
    df = df.assign(_bsf=bsf)
    out["auc_err"] = df.groupby(KEYS, sort=False)["_bsf"].mean().values
    out["regret"] = df.groupby(KEYS, sort=False)["_bsf"].last().values

    # time of first hit. inf if none — using the median handles censoring naturally (§6).
    hit_t = (df[df["is_hit"]].groupby(KEYS, sort=False)["t"].min()
             .reindex(pd.MultiIndex.from_frame(out[KEYS])).values)
    out["t_hit"] = np.where(np.isnan(hit_t.astype(float)), np.inf, hit_t)

    rng = {d: _y_range(d) for d in out["dataset"].unique()}
    scale = out["dataset"].map(rng).astype(float)
    out["auc_err_n"] = out["auc_err"] / scale
    out["regret_n"] = out["regret"] / scale
    return out


def paired(df: pd.DataFrame, a: str = "coherence", b: str = "base",
           metric: str = "auc_err_n") -> pd.DataFrame:
    """Pair two variants on (ds, q, surrogate, acq, seed) and form the difference.

    Because D₀ and CRN are determined independently of the variant (gates 8·10),
    each pair shares **the same initial design and the same random numbers**.
    So the pairwise differences can go straight into the test.
    """
    cell = ["dataset", "family", "target_q", "surrogate", "acquisition", "seed"]
    A = df[df["variant"] == a].set_index(cell)[metric]
    B = df[df["variant"] == b].set_index(cell)[metric]
    idx = A.index.intersection(B.index)
    return pd.DataFrame({"delta": (A.loc[idx] - B.loc[idx]).values},
                        index=idx).reset_index()
