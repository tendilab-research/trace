"""Data loading (plan §1.6).

The output is always a single `Pool`: `X` (n × d, float64), `y` (n,), `RecipeMeta`.
Policy code never receives `y` directly — it is wrapped in `OracleGuard` (gate 9).

Preprocessing follows §1.6 exactly:
  1. Load (Liang=with header / Olympus=no header, column order from config.json)
  2. G1  add balance component (AgNP)
  3. G7  aggregate by unique X, y = replicate **mean**
  4. G14 decide evaluation space (all y>0 AND skew>1 → log10). Current suite is all identity
  5. Validate (RecipeMeta.validate)

Composition renormalization (G2) is done **once at pool construction** (§1.6-2 revision).

Originally the raw X was left untouched and normalized only right before
`actions.decompose`. Then `apply(ref, all actions)` produced a **normalized**
candidate that disagreed with the original pool point. On p3ht the composition
sums range 99.89–100.10, giving a gap of up to 8.8e-02 (composition units).

V4 (acquisition decomposition, §8) makes this fatal. That design claims

    Σ_k φ_k = acq(x) − acq(ref)      ← **exactly**

but if the reconstructed x differs from the original x, the left side
decomposes the value at the reconstructed point and the identity degrades to
an approximation (measured error 2.05e-01). The other 7 datasets have exact
sums and were unaffected, so this went unnoticed until now.

The normalization itself is harmless — the deviation is rounding in the raw
data (≤0.1%), and compositions must by definition sum to `comp_total`.
Gates 2 and 16 check this property.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from .meta import BALANCE_COL, DATASETS, DatasetSpec, RecipeMeta

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Pool:
    """One offline pool. This object is the entirety of one (dataset) experiment."""

    X: np.ndarray                 # (n, d) float64 — surrogate input, original scale
    y: np.ndarray                 # (n,)   float64 — after evaluation-space transform
    meta: RecipeMeta
    y_raw: np.ndarray             # (n,)   original y before transform (for records)
    eval_space: str               # 'identity' | 'log10'  (G14)

    @property
    def n(self) -> int:
        return len(self.y)


# ══════════════════════════════════════════════════════════════════════════
#  Raw readers
# ══════════════════════════════════════════════════════════════════════════
def _read_liang(spec: DatasetSpec, root: Path) -> pd.DataFrame:
    return pd.read_csv(root / spec.path, encoding=spec.encoding)


def _read_olympus(spec: DatasetSpec, root: Path) -> pd.DataFrame:
    """Olympus `data.csv` has no header. Column order is `parameters` (in
    order) from config.json followed by `measurements` (§1.6-8)."""
    d = root / spec.path
    text = (d / "config.json").read_text()
    cfg = json.loads(re.sub(r"(?m)//.*$", "", text))    # config contains // comments
    names = [p["name"] for p in cfg["parameters"]] + [m["name"] for m in cfg["measurements"]]
    return pd.read_csv(d / "data.csv", header=None, names=names)


_READERS = {"liang": _read_liang, "olympus": _read_olympus}


# ══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════════
def _add_balance(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, list[str]]:
    """G1 — add an implicit balance component to pools whose composition sum is not constant.

    AgNP's four flow fractions sum to anywhere in 32.92–89.24. Making the
    remainder (solvent/diluent) an explicit component restores the simplex,
    which is required for paired-transfer actions to be defined. Without this
    there would be no mixed composition+process pool at all (§1.5).
    """
    comp = list(spec.comp_cols)
    if spec.balance_from is None:
        return df, comp
    bal = spec.balance_from - df[comp].sum(axis=1)
    assert bal.min() > -1e-9, (
        f"{spec.name}: negative balance component ({bal.min():.4g}). "
        f"balance_from={spec.balance_from} is wrong.")
    out = df.copy()
    out[BALANCE_COL] = bal.clip(lower=0.0)
    return out, comp + [BALANCE_COL]


def _aggregate_replicates(df: pd.DataFrame, x_cols: list[str], y_col: str
                          ) -> tuple[pd.DataFrame, int]:
    """G7 — collapse replicate measurements of identical X into their mean.

    Skipping this overestimates pool size by 20x (AgNP 3295 vs 164) and
    creates meaningless episodes that 'explore' the same composition repeatedly.
    """
    n_before = len(df)
    g = df.groupby(x_cols, sort=False, as_index=False)[y_col].mean()
    reps = int(round(n_before / max(len(g), 1)))
    return g, reps


def _qualifies_log(y: np.ndarray) -> bool:
    """G14 original rule — all y>0 and skew>1."""
    return bool((y > 0).all() and skew(y) > 1.0)


@functools.lru_cache(maxsize=None)
def _family_eval_space(family: str, root_key: str) -> str:
    """G14 — the evaluation space is decided **unanimously per family**.

    Applying the rule per dataset splits the OER family (measured:
    oer_plate_3496 skew=1.703 → log10, oer_plate_4098 skew=0.661 → identity).
    The two plates **share the same X pool and differ only in the y surface**,
    which is central to the design (§1.3); if their evaluation spaces
    diverge, the family-level bootstrap of §6.3 and the interpretation as
    "replicates with fully controlled composition space" collapse together.

    So log10 is used only when **every** member of the family qualifies.
    Otherwise fall to the conservative side (identity).

    Measured: only photo_pce10 is log10, the other 7 are identity.
    (The "all identity" in plan §1.6-4 was inaccurate.)
    """
    root = Path(root_key)
    members = [n for n, sp in DATASETS.items() if sp.family == family]
    return "log10" if all(_qualifies_log(_raw_y(n, root)) for n in members) else "identity"


@functools.lru_cache(maxsize=None)
def _raw_y(name: str, root: Path) -> np.ndarray:
    """Raw y, processed only through aggregation. Used solely for the evaluation-space decision."""
    spec = DATASETS[name]
    df = _READERS[spec.loader](spec, root)
    df, comp_cols = _add_balance(df, spec)
    x_cols = comp_cols + list(spec.proc_cols)
    df, _ = _aggregate_replicates(df, x_cols, spec.y_col)
    return df[spec.y_col].to_numpy(dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
@functools.lru_cache(maxsize=None)
def load(name: str, root: str | Path | None = None) -> Pool:
    """dataset name → `Pool`. Cached within a process."""
    spec = DATASETS[name]
    root = Path(root) if root is not None else DATA_ROOT

    df = _READERS[spec.loader](spec, root)
    df, comp_cols = _add_balance(df, spec)

    proc_cols = list(spec.proc_cols)
    x_cols = comp_cols + proc_cols
    missing = [c for c in x_cols + [spec.y_col] if c not in df.columns]
    assert not missing, f"{name}: missing columns {missing}. Available: {list(df.columns)}"

    df, reps = _aggregate_replicates(df, x_cols, spec.y_col)

    X = df[x_cols].to_numpy(dtype=np.float64)
    if comp_cols:
        # G2 — scale compositions to exactly comp_total. Done **once, here only**.
        ci = np.arange(len(comp_cols))
        cs = X[:, ci].sum(axis=1, keepdims=True)
        X[:, ci] = np.where(cs > 0.0, X[:, ci] * (spec.comp_total / cs), X[:, ci])
    y_raw = df[spec.y_col].to_numpy(dtype=np.float64)

    space = _family_eval_space(spec.family, str(root))
    y = np.log10(y_raw) if space == "log10" else y_raw

    meta = RecipeMeta(
        name=name, family=spec.family, columns=tuple(x_cols),
        comp_idx=np.arange(len(comp_cols), dtype=np.int64),
        proc_idx=np.arange(len(comp_cols), len(x_cols), dtype=np.int64),
        comp_total=spec.comp_total, y_name=spec.y_col,
        y_direction=spec.y_direction,
        has_balance=spec.balance_from is not None,
        comp_rtol=spec.comp_rtol, n_replicates=reps,
    )
    meta.validate(X)

    assert np.isfinite(X).all(), f"{name}: NaN/Inf in X"
    assert np.isfinite(y).all(), f"{name}: NaN/Inf in y"
    assert len(np.unique(X, axis=0)) == len(X), f"{name}: duplicate X remains after aggregation"

    return Pool(X=X, y=y, meta=meta, y_raw=y_raw, eval_space=space)


def summary() -> pd.DataFrame:
    """Load every dataset and build a one-page summary table (gate 0)."""
    rows = []
    for name in DATASETS:
        p = load(name)
        rows.append({
            "dataset": name, "family": p.meta.family, "N": p.n, "d": p.meta.d,
            "comp": p.meta.n_comp, "proc": p.meta.n_proc,
            "balance": p.meta.has_balance, "reps": p.meta.n_replicates,
            "comp_total": p.meta.comp_total, "eval_space": p.eval_space,
            "y_min": p.y.min(), "y_med": float(np.median(p.y)), "y_max": p.y.max(),
            "skew": float(skew(p.y_raw)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(summary().to_string(index=False))
