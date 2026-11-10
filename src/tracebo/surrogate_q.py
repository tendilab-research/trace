"""Surrogate quality — the `surrogate_q` table (plan §6.1).

**Why it is needed.** The largest confound in today's measurements was that
"base (acquisition argmax) is bad to a different degree per cell". In the RF
cell even random reranking beat base by a wide margin (gate 12a failed), so the
H1 gain in that cell cannot be attributed to the method.

To tell whether that difference comes from **surrogate quality** or from the
shape of the acquisition function, we must measure posterior calibration.
That is what this table does.

- `mae/rmse`  point-estimate accuracy
- `nll`       quality of the whole distribution. Same point estimate but overconfident -> penalized here
- `coverage`  do the 50%/90% intervals actually cover that fraction. **Low if overconfident**
- `calibration_err`  deviation of PIT from uniform. 0 = perfectly calibrated

Uses oracle y, so this is **not a policy module** — it must not be in gate 9's
POLICY_MODULES and is called only after selection is done.
"""

from __future__ import annotations

import numpy as np

Q_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def surrogate_quality(Y: np.ndarray, y_true: np.ndarray) -> dict:
    """Posterior quality for one iteration.

    `Y` is the (S, n) posterior sample already drawn by
    `acquisitions(..., return_samples=True)`. **We do not resample here** — one
    extra surrogate call per iteration would double the TabPFN lane cost.

    Everything is sample-based because the purpose of this table is to compare
    GP (continuous posterior) and TabPFN (discrete bar distribution) on the same axis.
    """
    y = np.asarray(y_true, dtype=float)
    Y = np.asarray(Y, dtype=float)                              # (S, n)
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    resid = mu - y

    # PIT = fraction of posterior samples below the true value. U(0,1) if perfectly calibrated.
    pit = (Y < y[None, :]).mean(axis=0)
    # Distance from uniform (simple Cramér–von Mises-style proxy)
    ps = np.sort(pit)
    unif = (np.arange(len(ps)) + 0.5) / len(ps)
    calib = float(np.abs(ps - unif).mean())

    out = {
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        # Gaussian-approximation NLL. Zero sample variance = overconfidence, so floor it.
        "nll": float(0.5 * np.mean(np.log(2 * np.pi * np.maximum(sd, 1e-9) ** 2)
                                   + (resid / np.maximum(sd, 1e-9)) ** 2)),
        "calibration_err": calib,
        "posterior_sd_mean": float(sd.mean()),
    }
    for lvl, tag in ((0.50, "50"), (0.90, "90")):
        lo = np.quantile(Y, (1 - lvl) / 2, axis=0)
        hi = np.quantile(Y, 1 - (1 - lvl) / 2, axis=0)
        out[f"coverage_{tag}"] = float(np.mean((y >= lo) & (y <= hi)))
    return out
