"""Common surrogate interface (plan §4.7).

Policy and acquisition see only these three methods.

- `fit(X, y)`
- `moments(Z) -> (mu, sd) | None` — closed-form posterior moments. None if unavailable.
- `sample(Z, S, rng) -> (S, n)`   — posterior samples. **rng is mandatory.**

`sample` takes rng as an argument because of G10 (common random numbers).
base-TS and TRACE-TS must see **identical** posterior samples from the same
state to claim "reranking won" (gate 10). A model using internal global
randomness breaks this comparison.

`moments` was allowed to return None originally because of TabPFN. Now
**TabPFN implements it too** — the 1st/2nd moments of the bar distribution are
closed-form since it is piecewise uniform. Reporting two moments is not the
same as assuming normality. V5's channel decomposition (`A₀ = lim_{σ→0} A`)
uses only μ, so it is independent of posterior shape, which lets us compare
the same acquisition across surrogates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """Base for all surrogates. Input standardization is handled here, once."""

    name: str = "base"

    def __init__(self) -> None:
        self._mx: np.ndarray | None = None
        self._sx: np.ndarray | None = None
        self._my: float = 0.0
        self._sy: float = 1.0
        self._fitted = False

    # --- Standardization ---------------------------------------------------
    def _standardize_fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._mx = X.mean(axis=0)
        sx = X.std(axis=0)
        # Constant columns (e.g. a process variable not yet varied with few observations) get 1.
        self._sx = np.where(sx < 1e-12, 1.0, sx)
        self._my = float(y.mean())
        sy = float(y.std())
        self._sy = sy if sy > 1e-12 else 1.0
        return (X - self._mx) / self._sx, (y - self._my) / self._sy

    def _standardize_x(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self._mx) / self._sx

    def _unscale_mu(self, mu: np.ndarray) -> np.ndarray:
        return mu * self._sy + self._my

    def _unscale_sd(self, sd: np.ndarray) -> np.ndarray:
        return sd * self._sy

    # --- To be implemented by subclasses -----------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...

    @abstractmethod
    def sample(self, Z: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray: ...

    def moments(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Closed-form posterior moments. Models without them return None."""
        return None

    def _check(self) -> None:
        assert self._fitted, f"{self.name}: fit must be called first"
