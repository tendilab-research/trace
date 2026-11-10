"""GP surrogate — ARD Matérn-5/2 (spec §7).

Gives a Gaussian posterior, so `moments` is implemented. The target utility
can then be computed in closed form without samples (§4.7), so even a large
coalition matrix costs almost nothing. This is why the GP lane stays on CPU.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import Surrogate

MIN_SD = 1e-9


class GPSurrogate(Surrogate):
    name = "GP"

    def __init__(self, n_restarts: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.n_restarts = n_restarts
        self.seed = seed
        self._gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        Xs, ys = self._standardize_fit(X, y)
        d = Xs.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(d),                 # ★ ARD
                           length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(1e-4, (1e-8, 1e1)))
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False,
            n_restarts_optimizer=self.n_restarts, random_state=self.seed)
        with warnings.catch_warnings():
            # With n of 5–30, hyperparameters often hit the bounds.
            # Predictions remain valid, so do not flood the log with warnings.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._gp.fit(Xs, ys)
        self._fitted = True
        return self

    def moments(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check()
        mu, sd = self._gp.predict(self._standardize_x(Z), return_std=True)
        return self._unscale_mu(mu), np.maximum(self._unscale_sd(sd), MIN_SD)

    def sample(self, Z: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
        """Independent samples from the marginal posterior. **Not** the joint posterior.

        The acquisition (§4.8) uses only per-candidate marginals, and Thompson
        sampling just takes the argmax over the pool, so no joint covariance is
        needed. Avoiding the n×n covariance keeps cost linear even for the
        coalition matrix (20k rows).
        """
        mu, sd = self.moments(Z)
        return mu[None, :] + sd[None, :] * rng.standard_normal((S, len(mu)))
