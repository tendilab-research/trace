"""TabPFN surrogate (GPU lane).

**The model is pinned to v2.** Use the latest package (8.x) but explicitly
request `ModelVersion.V2` weights. Rationale:

- The spec cites TabPFN **v2** (Hollmann 2025, Nature
  doi:10.1038/s41586-024-08328-6). What changes is the code, not the model.
- **License**: only the v2 weights are under the Prior Labs License (Apache 2.0
  + attribution). v2.5 / v2.6 / v3 are non-commercial, which constrains the paper.
- Staying on package 2.0.x would lose a year of bug fixes and is unverified
  with current torch/numpy combinations.

TabPFN yields a discrete bar distribution, not a Gaussian posterior. So
`moments` **returns None** — `E|Y−y*|` cannot be written in closed form from
mean/std (§4.7). Utility goes through the sample mean.
"""

from __future__ import annotations

import numpy as np

from .base import Surrogate
from .device import resolve_device

MIN_SD = 1e-9             # same floor as gp.py/rf.py — prevents σ=0
N_ESTIMATORS = 4          # gate 13 compares 4 vs 8 and approves
PREDICT_CHUNK = 2048      # batches (chunk × 5000 bins) float64 ≈ 82MB


def _make_regressor(device: str, n_estimators: int, seed: int):
    """Explicitly request v2 weights. API differs by package major version."""
    from tabpfn import TabPFNRegressor

    try:                                    # 8.x / 6.x — model version selectable
        from tabpfn.constants import ModelVersion

        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device=device,
            n_estimators=n_estimators, random_state=seed)
    except ImportError:                     # 2.0.x — v2 is the only model
        return TabPFNRegressor(device=device, n_estimators=n_estimators,
                               random_state=seed)


class TabPFNSurrogate(Surrogate):
    name = "TabPFN"

    def __init__(self, n_estimators: int = N_ESTIMATORS, seed: int = 0,
                 device: str | None = None) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.seed = seed
        self.device = device or resolve_device()
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNSurrogate":
        # TabPFN is an in-context learner and standardizes internally. Still,
        # run the base scaler to keep the input convention consistent with other surrogates.
        Xs, ys = self._standardize_fit(X, y)
        self._model = _make_regressor(self.device, self.n_estimators, self.seed)
        self._model.fit(Xs, ys)
        self._fitted = True
        return self

    def _predict_full(self, Z: np.ndarray) -> dict:
        """Fetch full predictions (including quantiles) in chunks."""
        Zs = self._standardize_x(Z)
        outs = []
        for i in range(0, len(Zs), PREDICT_CHUNK):
            outs.append(self._model.predict(Zs[i:i + PREDICT_CHUNK],
                                            output_type="full"))
        return outs

    def sample(self, Z: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
        """S posterior samples via **inverse-transform sampling** of the bar distribution.

        The library's `crit.icdf` accepts `left_prob` only as a **scalar**, and
        `crit.sample` calls icdf in a per-row Python loop using `torch.rand`
        (global RNG). None of these work here:

        - Calling the scalar icdf S times makes every row share the **same**
          uniform draw -> perfectly correlated samples. Thompson sampling's
          argmax then always equals the posterior-mean argmax, and TS degenerates.
        - The global RNG breaks G10 (common random numbers). base-TS and
          TRACE-TS must see identical samples for gate 10 to hold.
        - The per-row loop is 12,470 rows × 64 samples = 800k calls per iteration.

        So we vectorize the **same formula** as `icdf` in numpy.
        """
        self._check()
        out = np.empty((S, len(Z)), dtype=np.float64)
        start = 0
        for res in self._predict_full(Z):
            logits = res["logits"].detach().cpu().numpy().astype(np.float64)
            borders = res["criterion"].borders.detach().cpu().numpy().astype(np.float64)
            n = logits.shape[0]
            u = rng.random((S, n))                       # ★ independent uniforms per row
            out[:, start:start + n] = _icdf_batch(logits, borders, u)
            start += n
        return self._unscale_mu(out)

    def moments(self, Z: np.ndarray):
        """**Exact** 1st/2nd moments of the bar distribution.

        The posterior is piecewise uniform: probability `p_i` spread uniformly
        over the interval `[b_i, b_{i+1})`. Hence the moments are closed-form:

            E[Y]   = Σ p_i · (b_i + b_{i+1}) / 2
            E[Y²]  = Σ p_i · (b_i² + b_i b_{i+1} + b_{i+1}²) / 3

        **This is not a normality assumption.** We only report the two moments
        exactly; the acquisition (§4.8) applies Gaussian-form formulas on top.
        The scope of that approximation is stated in paper §3 — to compare the
        **same acquisition** across surrogates, the acquisition must be fixed
        as a functional of (μ, σ).

        Previously this returned None, which blocked the V4/V5 paths entirely.
        There was no reason to: `A₀ = lim_{σ→0} A` uses only μ, so it is
        defined **for any posterior shape**, and the shared-baseline proposition
        requires nothing beyond the posterior collapsing to a point
        (distribution-free).
        """
        self._check()
        n = len(Z)
        mu = np.empty(n, dtype=np.float64)
        m2 = np.empty(n, dtype=np.float64)
        start = 0
        for res in self._predict_full(Z):
            logits = res["logits"].detach().cpu().numpy().astype(np.float64)
            borders = res["criterion"].borders.detach().cpu().numpy().astype(np.float64)
            # softmax — subtract the per-row max to prevent overflow
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            pr = e / e.sum(axis=1, keepdims=True)
            lo, hi = borders[:-1], borders[1:]
            k = logits.shape[0]
            mu[start:start + k] = pr @ (0.5 * (lo + hi))
            m2[start:start + k] = pr @ ((lo * lo + lo * hi + hi * hi) / 3.0)
            start += k
        var = np.maximum(m2 - mu * mu, 0.0)
        return self._unscale_mu(mu), np.maximum(self._unscale_sd(np.sqrt(var)),
                                                MIN_SD)


def _icdf_batch(logits: np.ndarray, borders: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of the bar distribution for (S, n) uniforms in one shot.

    `logits` (n, B), `borders` (B+1,), `u` (S, n) → (S, n).

    numpy has no per-row `searchsorted`. Adding `2i` to row i's cumulative
    probabilities places its values in `[2i, 2i+1]`, making the whole array
    globally sorted; applying the same offset to the queries finishes with
    **a single** searchsorted.
    """
    n, B = logits.shape
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0                                  # fix accumulated rounding

    off = (np.arange(n, dtype=np.float64) * 2.0)[:, None]
    idx = np.searchsorted((cum + off).ravel(), (u + off.T).ravel())
    idx = idx.reshape(u.shape) - np.arange(n)[None, :] * B
    np.clip(idx, 0, B - 1, out=idx)

    rows = np.broadcast_to(np.arange(n)[None, :], u.shape)
    cum_left = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    p = np.maximum(probs[rows, idx], 1e-30)           # guard against empty bins
    lo, hi = borders[idx], borders[idx + 1]
    return lo + (hi - lo) * np.clip((u - cum_left) / p, 0.0, 1.0)
