"""Recipe -> action decomposition and re-application (spec §6.2, plan §4.5).

There are only two kinds of action.

- **comp** — paired transfer `δ_{j→i} = d_j·u_i / M`.
  Subtract δ from component j of the reference recipe and add it to component i.
  `d = (c_ref − c_cand)₊`, `u = (c_cand − c_ref)₊`, `M = Σd`.
- **proc** — set process variable v to the candidate's value.

Three properties of this design — all checked by gates 1 and 2:

1. **Exact reconstruction** — applying all actions yields exactly the candidate.
   `Σ_i δ_{j→i} = d_j·(Σu)/M = d_j` (since Σd = Σu = M).
2. **Every subset preserves the simplex** — each comp action moves −δ and +δ as
   a pair, so **any subset** keeps the sum invariant and no component goes negative.
   Shapley needs the utility of every subset; without this property we would
   feed the surrogate recipes outside the coordinate system.
3. **Order-independent** — comp actions are additions on distinct (j,i) pairs
   and proc actions are assignments, so they commute. Hence `apply` takes a
   **set**, not a permutation. This is why Shapley need not enumerate permutations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meta import RecipeMeta

TOL = 1e-9


@dataclass(frozen=True)
class Action:
    kind: str                  # 'comp' | 'proc'
    donor: int = -1            # comp: donor component (position within comp_idx)
    receiver: int = -1         # comp: receiving component
    delta: float = 0.0
    var: int = -1              # proc: column index in X
    value: float = 0.0

    def label(self, meta: RecipeMeta) -> str:
        if self.kind == "comp":
            return (f"{meta.columns[meta.comp_idx[self.donor]]}"
                    f"→{meta.columns[meta.comp_idx[self.receiver]]}"
                    f" ({self.delta:+.4g})")
        return f"{meta.columns[self.var]}={self.value:.4g}"


def renorm(c: np.ndarray, total: float) -> np.ndarray:
    """G2 — renormalize the composition to exactly `total`.

    Raw P3HT sums range 99.89–100.10. Left as-is, gate 2's sum-preservation
    check (atol 1e-9) fails. Surrogate inputs keep the raw values; we normalize
    **only right before action decomposition** (§1.6-2).
    """
    s = c.sum()
    return c if s == 0.0 else c * (total / s)


def decompose(ref: np.ndarray, cand: np.ndarray, meta: RecipeMeta) -> list[Action]:
    """Action set connecting reference -> candidate. Deterministic, fixed order."""
    acts: list[Action] = []

    if meta.n_comp:
        c_r = renorm(ref[meta.comp_idx], meta.comp_total)
        c_c = renorm(cand[meta.comp_idx], meta.comp_total)
        d = np.maximum(c_r - c_c, 0.0)
        u = np.maximum(c_c - c_r, 0.0)
        M = d.sum()
        if M > TOL:
            for j in np.flatnonzero(d > TOL):
                for i in np.flatnonzero(u > TOL):
                    acts.append(Action("comp", donor=int(j), receiver=int(i),
                                       delta=float(d[j] * u[i] / M)))

    for v in meta.proc_idx:
        if abs(ref[v] - cand[v]) > TOL:
            acts.append(Action("proc", var=int(v), value=float(cand[v])))

    return acts


def apply(ref: np.ndarray, subset, meta: RecipeMeta) -> np.ndarray:
    """Apply a **subset** of actions to the reference recipe (order-independent)."""
    z = ref.copy()
    if meta.n_comp:
        z[meta.comp_idx] = renorm(z[meta.comp_idx], meta.comp_total)
    for a in subset:
        if a.kind == "comp":
            z[meta.comp_idx[a.donor]] -= a.delta
            z[meta.comp_idx[a.receiver]] += a.delta
        else:
            z[a.var] = a.value
    if meta.n_comp:
        # Paired transfers are algebraically exact, but accumulated float error
        # can leave values like −1e-17. Clamp so we stay inside the coordinate system.
        np.clip(z[meta.comp_idx], 0.0, None, out=z[meta.comp_idx])
    return z
