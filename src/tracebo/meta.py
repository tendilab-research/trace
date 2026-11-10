"""Dataset registry and `RecipeMeta`.

Split into two layers.

- `DatasetSpec` — **declarative**. Where the raw file lives and which columns
  are composition vs. process. Meant for human review; consumed by `data.py`.
- `RecipeMeta`  — **runtime**. Column *indices* into an already-loaded `X`
  matrix. `actions.py` / `shapley.py` see only this.

Why the split: Liang CSVs have headers, Olympus data.csv does not. Mixing
column names and column indices in one object leaks that difference
throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOL = 1e-9


# ══════════════════════════════════════════════════════════════════════════
#  Runtime meta
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RecipeMeta:
    """Column role assignment for a loaded `X` (n × d).

    `comp_idx` is the set of composition columns **including the balance
    component** (G1). Hence `X[:, comp_idx].sum(1) == comp_total` must hold
    for every row, which gate 2 checks.
    """

    name: str
    family: str
    columns: tuple[str, ...]
    comp_idx: np.ndarray            # int64, composition columns (incl. balance)
    proc_idx: np.ndarray            # int64, process columns
    comp_total: float               # composition sum. 0.0 if no composition
    y_name: str
    y_direction: str                # 'min' | 'max' — for the record only (target-oriented, irrelevant to policies)
    has_balance: bool = False       # G1: is the last entry of comp_idx an implicit balance component
    comp_rtol: float = 1e-6         # relative tolerance on the raw composition sum (G2: P3HT is loose)
    n_replicates: int = 1           # G7: number of replicates averaged in the raw data (median)

    @property
    def d(self) -> int:
        return len(self.columns)

    @property
    def n_comp(self) -> int:
        return len(self.comp_idx)

    @property
    def n_proc(self) -> int:
        return len(self.proc_idx)

    def validate(self, X: np.ndarray) -> None:
        """Data side of gate 2. Called once right after loading."""
        assert X.ndim == 2 and X.shape[1] == self.d, \
            f"{self.name}: X shape {X.shape} != (n, {self.d})"
        roles = np.concatenate([self.comp_idx, self.proc_idx])
        assert sorted(roles.tolist()) == list(range(self.d)), \
            f"{self.name}: every column must have exactly one role (comp∪proc == all)"
        if self.n_comp:
            tot = X[:, self.comp_idx].sum(axis=1)
            dev = np.abs(tot - self.comp_total).max()
            # The raw data may be loose (G2). Strict sum preservation of
            # coalition recipes is checked separately by gate 2 after actions.renorm.
            assert dev <= self.comp_rtol * max(1.0, self.comp_total), \
                f"{self.name}: composition sum is not {self.comp_total} (max deviation {dev:.3g})"
            assert X[:, self.comp_idx].min() >= -TOL, \
                f"{self.name}: negative composition component"


# ══════════════════════════════════════════════════════════════════════════
#  Declarative registry
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    loader: str                     # 'liang' | 'olympus'
    path: str                       # relative to data/
    comp_cols: tuple[str, ...] = ()
    proc_cols: tuple[str, ...] = ()
    comp_total: float = 0.0
    balance_from: float | None = None   # G1: balance component = this total minus the composition sum
    y_col: str = ""
    y_direction: str = "min"
    encoding: str = "utf-8"
    comp_rtol: float = 1e-6
    note: str = ""


# Final suite of 8 (plan §1.2). Perovskite excluded because m≤2 (G4).
DATASETS: dict[str, DatasetSpec] = {
    # ── OER: 6-component simplex, m_max=9. Primary. 4 plates share the X pool ──
    "oer_plate_3496": DatasetSpec(
        name="oer_plate_3496", family="OER", loader="olympus",
        path="raw/olympus/dataset_oer_plate_3496",
        comp_cols=("ni_load", "fe_load", "co_load", "mn_load", "ce_load", "la_load"),
        comp_total=1.0, y_col="overpotential", y_direction="min",
        note="Rohr 2020. config.json constraints.parameters == 'simplex'.",
    ),
    "oer_plate_4098": DatasetSpec(
        name="oer_plate_4098", family="OER", loader="olympus",
        path="raw/olympus/dataset_oer_plate_4098",
        comp_cols=("sn_load", "sb_load", "co_load", "ca_load", "ni_load", "mn_load"),
        comp_total=1.0, y_col="overpotential", y_direction="min",
        note="Same X pool as 3496, different y surface. Grouped at the family level (§6.3).",
    ),

    # ── Composition ───────────────────────────────────────────────────────
    "p3ht": DatasetSpec(
        name="p3ht", family="P3HT", loader="liang",
        path="raw/liang/P3HT_dataset.csv",
        comp_cols=("P3HT content (%)", "D1 content (%)", "D2 content (%)",
                   "D6 content (%)", "D8 content (%)"),
        comp_total=100.0, y_col="Conductivity (measured) (S/cm)", y_direction="max",
        comp_rtol=2e-3,
        note=("G2: raw sums drift between 99.89–100.10. Surrogate inputs keep the "
              "raw values; renormalize to exactly 100 only right before action decomposition (§1.6-2)."),
    ),
    "photo_pce10": DatasetSpec(
        name="photo_pce10", family="Photo", loader="olympus",
        path="raw/olympus/dataset_photo_pce10",
        comp_cols=("mat_1", "mat_2", "mat_3", "mat_4"),
        comp_total=1.0, y_col="degradation", y_direction="min",
    ),

    # ── Mixed (composition + process). The only one among public pools → why G1 is required ──
    "agnp": DatasetSpec(
        name="agnp", family="AgNP", loader="liang",
        path="raw/liang/AgNP_dataset.csv",
        comp_cols=("QAgNO3(%)", "Qpva(%)", "Qtsc(%)", "Qseed(%)"),
        proc_cols=("Qtot(uL/min)",),
        comp_total=100.0, balance_from=100.0,
        y_col="loss", y_direction="min",
        note=("G1: the sum of the 4 flow fractions is not constant (32.92–89.24). "
              "The remainder (=100−sum) is treated as an implicit balance component to restore the simplex. "
              "This is the only mixed composition+process pool."),
    ),

    # ── Process only ──────────────────────────────────────────────────────
    "crossed_barrel": DatasetSpec(
        name="crossed_barrel", family="CB", loader="liang",
        path="raw/liang/Crossed barrel_dataset.csv",
        proc_cols=("n", "theta", "r", "t"),
        y_col="toughness", y_direction="max",
        note="G7: exactly triplicate (1800 → 600).",
    ),
    "hplc": DatasetSpec(
        name="hplc", family="hplc", loader="olympus",
        path="raw/olympus/dataset_hplc",
        proc_cols=("sample_loop", "additional_volume", "tubing_volume",
                   "sample_flow", "push_speed", "wait_time"),
        y_col="peak_area", y_direction="max",
        note="G7: 1386 → 1007.",
    ),
    "autoam": DatasetSpec(
        name="autoam", family="AutoAM", loader="liang",
        path="raw/liang/AutoAM_dataset.csv",
        proc_cols=("Prime Delay", "Print Speed",
                   "X Offset Correction", "Y Offset Correction"),
        y_col="Score", y_direction="max",
        note="No replicates (100/100).",
    ),
}

# Scope ladder by GPU count (§7.2). With 1–2 GPUs, reduce to 4 families.
SUITE_SMALL = ("oer_plate_3496", "p3ht", "agnp", "crossed_barrel")
SUITE_FULL = tuple(DATASETS)

BALANCE_COL = "_balance"


def suite(n_gpu: int) -> tuple[str, ...]:
    return SUITE_FULL if n_gpu >= 3 else SUITE_SMALL
