"""Oracle-leak prevention (gate 9).

Policy code **never receives** the `y` array directly. It only gets an
`OracleGuard`. Reading `y` of a not-yet-observed candidate raises immediately.

Why this is the most important of the 14 gates: an oracle leak is invisible at
the aggregation stage. TRACE simply looks very good, and the leak is found only
after the result is in the paper. Two layers block it: runtime guard + static
symbol check.
"""

from __future__ import annotations

import numpy as np


class OracleLeak(AssertionError):
    """Accessed y of an unobserved candidate."""


class OracleGuard:
    """The only object handed to policy code.

    - `observe(i)`   : observe candidate i and return its y. Only place the observed set grows.
    - `y_of(idx)`    : y of **already observed** candidates. Any unobserved index -> OracleLeak.
    - `reveal_for_metrics()` : full y. **Evaluation logger only.**
      If this symbol appears in a policy module, gate 9 (static check) fails.
    """

    __slots__ = ("_y", "_observed", "_order")

    def __init__(self, y: np.ndarray):
        self._y = np.asarray(y)
        self._observed: set[int] = set()
        self._order: list[int] = []

    # --- Available to policy code ------------------------------------------
    def observe(self, i: int) -> float:
        i = int(i)
        if i not in self._observed:
            self._observed.add(i)
            self._order.append(i)
        return float(self._y[i])

    def observe_many(self, idx) -> np.ndarray:
        return np.array([self.observe(i) for i in np.atleast_1d(idx)])

    def y_of(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        unseen = set(idx.tolist()) - self._observed
        if unseen:
            raise OracleLeak(
                f"ORACLE LEAK: requested y of unobserved candidates {sorted(unseen)[:8]} "
                f"({len(self._observed)} observed)."
            )
        return self._y[idx]

    @property
    def observed(self) -> np.ndarray:
        """Index array preserving observation order. Used to build D_t."""
        return np.asarray(self._order, dtype=np.int64)

    @property
    def n_observed(self) -> int:
        return len(self._observed)

    def unobserved(self, n_total: int) -> np.ndarray:
        mask = np.ones(n_total, dtype=bool)
        if self._order:
            mask[self.observed] = False
        return np.flatnonzero(mask)

    # --- Evaluation logger only --------------------------------------------
    def reveal_for_metrics(self) -> np.ndarray:
        """★ Evaluation logger only. Calling this from a policy module fails gate 9."""
        return self._y
