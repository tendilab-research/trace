"""Random Forest surrogate.

The posterior is approximated by the empirical distribution of the tree
ensemble. `moments` is the mean/std of tree predictions; `sample` draws trees
with replacement — so samples stay within the actual ensemble and, unlike GP,
no normality is assumed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import Surrogate

MIN_SD = 1e-9


class RFSurrogate(Surrogate):
    name = "RF"

    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.seed = seed
        self._rf: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFSurrogate":
        Xs, ys = self._standardize_fit(X, y)
        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed,
            n_jobs=1,                       # avoid overlapping with worker parallelism (§4.12)
            min_samples_leaf=1, bootstrap=True)
        self._rf.fit(Xs, ys)
        self._fitted = True
        return self

    def _tree_preds(self, Z: np.ndarray) -> np.ndarray:
        """(n_trees, n) per-tree predictions."""
        Zs = self._standardize_x(Z)
        return np.stack([t.predict(Zs) for t in self._rf.estimators_])

    def moments(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check()
        P = self._tree_preds(Z)
        return (self._unscale_mu(P.mean(axis=0)),
                np.maximum(self._unscale_sd(P.std(axis=0)), MIN_SD))

    def sample(self, Z: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
        self._check()
        P = self._tree_preds(Z)
        pick = rng.integers(0, P.shape[0], size=S)
        return self._unscale_mu(P[pick])
