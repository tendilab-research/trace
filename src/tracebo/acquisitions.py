"""Target-oriented utility and acquisition functions (plan §4.7–4.8).

**The goal is neither minimization nor maximization** — it is hitting y\\*.
Every utility is therefore based on `−|y − y*|`, and the standard BO
acquisitions are rebuilt on top of it.

G10 (common random numbers) governs this module's design. The RNG seed is
determined by `(seed, t, shard_id)` only — **the variant is not part of it**.
This is what lets base-TS and TRACE-TS in the same state see identical
posterior samples, so any difference can be attributed solely to reranking
(gate 10).
"""

from __future__ import annotations

from hashlib import blake2b

import numpy as np
from scipy.stats import norm

S_ACQ = 256          # posterior samples for acquisitions
S_UTIL = 64          # for coalition utility (only when no closed form exists)
ALPHA = 0.1          # Optimism quantile = 1 − α


def crn(seed: int, t: int, shard_id: str) -> np.random.Generator:
    """G10 — deterministic RNG source excluding the variant."""
    h = blake2b(f"{seed}|{t}|{shard_id}".encode(), digest_size=8).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h, "big") >> 1))


def target_utility(surr, Z: np.ndarray, y_star: float,
                   rng: np.random.Generator) -> np.ndarray:
    """`E[−|Y − y*|]`. The utility used by Shapley and **the most-called
    function** (12,470 rows per iteration on average).

    For a Gaussian posterior use the closed form. With `Y ~ N(μ,σ²)`,
    `t = (μ−y*)/σ`:
        `E|Y − y*| = σ·(2φ(t) + t·(2Φ(t) − 1))`
    No sampling is needed, so on the GP/RF lanes the cost stays negligible
    even for large coalition matrices. TabPFN has no Gaussian posterior, so
    it falls back to the sample mean.
    """
    mom = surr.moments(Z)
    if mom is not None:
        mu, sd = mom
        t = (mu - y_star) / sd
        return -sd * (2.0 * norm.pdf(t) + t * (2.0 * norm.cdf(t) - 1.0))
    return -np.abs(surr.sample(Z, S=S_UTIL, rng=rng) - y_star).mean(axis=0)


def acquisitions(surr, X_pool: np.ndarray, y_star: float, e_best: float,
                 rng: np.random.Generator, alpha: float = ALPHA,
                 return_samples: bool = False):
    """Compute all four acquisitions from **a single draw** of samples.

    - `EXP` pure exploitation — expected utility
    - `OPT` optimism         — (1−α) posterior quantile of the utility
    - `tEI` target EI        — expected improvement over the current best error `e_best`
    - `TS`  Thompson         — **one row** of posterior samples. Gate 10 checks
      that base/TRACE see identical samples, so no new rng may be created here.

    With `return_samples=True` returns `(dict, Y)`. The `surrogate_q` table
    **reuses** these samples — drawing separately would add one more surrogate
    call per iteration and double the TabPFN lane cost.
    """
    Y = surr.sample(X_pool, S=S_ACQ, rng=rng)          # (S, N)
    err = np.abs(Y - y_star)
    util = -err
    out = {
        "EXP": util.mean(axis=0),
        "OPT": np.quantile(util, 1.0 - alpha, axis=0),
        "tEI": np.maximum(0.0, e_best - err).mean(axis=0),
        "TS": util[0],
    }
    return (out, Y) if return_samples else out


ACQ_NAMES = ("EXP", "OPT", "tEI", "TS")


def ablation_seed(crn_key: str, t: int) -> int:
    """Seed stream dedicated to the Tier B `random` ablation.

    Kept separate from CRN. Consuming the `crn()` rng here would make
    posterior-sample consumption diverge across variants and break gate 10
    (TS fairness). This stream is used only by the `random` variant, so it
    does not affect the trajectories of other variants.
    """
    h = blake2b(f"ablation|{crn_key}|{t}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") >> 1


# ─────────────────────────────────────────────────────────────────────────
# V4 — deterministic path for using the acquisition **as the explained
# object itself** (plan §8)
#
# One design requirement: **acq(x) at selection time and acq(x) at
# explanation time must be exactly equal.** Only then does Shapley efficiency
#       Σ_k φ_k = acq(x) − acq(ref)
# hold, and we can say "the explanation exactly decomposes the reason we
# actually chose". `acquisitions()` above is a Monte Carlo over S=256
# posterior samples, so **its value depends on the batch** — acq(x) evaluated
# on the pool differs slightly from acq(x) evaluated on the coalition matrix.
# The identity then degrades to an approximation.
#
# So three of them switch to closed form / quadrature, and TS to **pointwise
# common random numbers**. As a side effect, the Monte Carlo noise (S=256)
# disappears from the acquisitions entirely.
# ─────────────────────────────────────────────────────────────────────────

_PT_DECIMALS = 12   # stabilizes point hashing (same rule as shapley._key)


def _exp_util(mu, sd, y_star):
    """E[−|Y − y*|],  Y ~ N(mu, sd²). Same closed form as `target_utility`."""
    t = (mu - y_star) / sd
    return -sd * (2.0 * norm.pdf(t) + t * (2.0 * norm.cdf(t) - 1.0))


def _abs_err_cdf(a, mu, sd, y_star):
    """P(|Y − y*| ≤ a)."""
    return (norm.cdf((y_star + a - mu) / sd)
            - norm.cdf((y_star - a - mu) / sd))


def _opt_util(mu, sd, y_star, alpha):
    """(1−α) quantile of −|Y−y*| = −(α quantile of |Y−y*|).

    The CDF of |Y−y*| above is monotone increasing, so solve by bisection.
    60 vectorized iterations reach double-precision limits (the interval
    halves each step).
    """
    lo = np.zeros_like(mu)
    hi = np.abs(mu - y_star) + 10.0 * sd
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        go = _abs_err_cdf(mid, mu, sd, y_star) < alpha
        lo = np.where(go, mid, lo)
        hi = np.where(go, hi, mid)
    return -0.5 * (lo + hi)


def _tei_util(mu, sd, y_star, e_best):
    """`E[max(0, e_best − |Y−y*|)]` — **closed form**.

        max(0, e−|d|)·1{|d|<e},  d = Y − y* ~ N(m, σ²),  m = μ − y*
        tEI = e·P(|d|<e) − E[|d|·1{|d|<e}]
        E[|d|·1{|d|<e}] = g(m) + g(−m),
            g(m) = ∫₀^e t·N(t; m, σ) dt
                 = m[Φ((e−m)/σ) − Φ(−m/σ)] + σ[φ(m/σ) − φ((e−m)/σ)]

    Initially this used the layer-cake integral `∫₀^{e} P(|Y−y*| ≤ u) du`
    with 64-point Gauss–Legendre. In the usual regime it matches the closed
    form to 1e-16, but **as σ shrinks the integrand becomes a step and the
    error stalls at 6e-3** — unusable, since the σ→0 limit (A₀ of V5) is
    exactly the regime of interest. The closed form converges exactly to
    `max(0, e−|m|)` as σ→0 (gate 20).
    """
    if e_best <= 0:
        return np.zeros_like(mu)
    m = mu - y_star

    def g(mm):
        a = -mm / sd
        b = (e_best - mm) / sd
        return (mm * (norm.cdf(b) - norm.cdf(a))
                + sd * (norm.pdf(a) - norm.pdf(b)))

    P = norm.cdf((e_best - m) / sd) + norm.cdf((e_best + m) / sd) - 1.0
    return e_best * P - (g(m) + g(-m))


def _pointwise_normal(Z: np.ndarray, key: str) -> np.ndarray:
    """Per-point fixed standard-normal draws — **invariant to batch composition**.

    An ordinary `rng.standard_normal(n)` ties values to the number and order
    of rows, so a TS sample drawn on the pool and one drawn on the coalition
    matrix give different values for the same point, breaking the identity.
    Hashing the coordinates binds the draw to the point itself, so the value
    is the same from whichever batch it is called — an extension of Thompson
    sampling's common random numbers (G10) into coordinate space.
    """
    out = np.empty(len(Z), dtype=np.float64)
    kb = key.encode()
    for i, row in enumerate(np.round(Z, _PT_DECIMALS)):
        h = blake2b(row.tobytes() + kb, digest_size=8).digest()
        u = (int.from_bytes(h, "big") >> 11) / float(1 << 53)   # (0,1)
        out[i] = norm.ppf(min(max(u, 1e-12), 1 - 1e-12))
    return out


def _moments(surr, Z: np.ndarray):
    """Models without `moments` (TabPFN) fail explicitly — the posterior is
    not Gaussian, so no closed form exists."""
    mom = surr.moments(Z)
    if mom is None:
        raise NotImplementedError(
            f"{surr.name}: deterministic acq requires moments (V4 path)")
    return mom


LAMBDA_CB = 1.0           # optimism coefficient of CB. Corresponds to λ in ShapleyBO.


def _cb_util(mu, sd, y_star, lam=LAMBDA_CB):
    """`CB = −|μ−y*| + λσ` — an **additive** acquisition.

    This is exactly the form ShapleyBO (Rodemann 2024) decomposes: the two
    terms are already written out, so applying Shapley linearity term by term
    is all it takes. It is included to show, by execution rather than
    rhetoric, that our definition **contains it as a special case** —
    `acq_plugin("CB") = −|μ−y*|` and `U = λσ`, so the `lim_{σ→0}` definition
    recovers exactly the two terms they split by hand.
    """
    return -np.abs(mu - y_star) + lam * sd


def _from_moments(mu, sd, Z, y_star, e_best, name, crn_key, alpha):
    if name == "CB":
        return _cb_util(mu, sd, y_star)
    if name == "EXP":
        return _exp_util(mu, sd, y_star)
    if name == "OPT":
        return _opt_util(mu, sd, y_star, alpha)
    if name == "tEI":
        return _tei_util(mu, sd, y_star, e_best)
    if name == "TS":
        return -np.abs(mu + sd * _pointwise_normal(Z, crn_key) - y_star)
    raise ValueError(f"unknown acquisition: {name}")


def acq_values(surr, Z: np.ndarray, y_star: float, e_best: float,
               name: str, crn_key: str, alpha: float = ALPHA) -> np.ndarray:
    """Evaluate the acquisition **deterministically** — same point, same value.

    Plug directly into `attribute_batch(..., utility_fn=...)` and Shapley
    decomposes the acquisition. By efficiency `Σφ_k = acq(x) − acq(ref)`,
    an exact action decomposition of **the very value that made the
    selection** (gate 16).

    Models without `moments` (TabPFN) are not supported — their posterior is
    not Gaussian, so no closed form exists. They fail explicitly.
    """
    mu, sd = _moments(surr, Z)
    return _from_moments(mu, sd, Z, y_star, e_best, name, crn_key, alpha)


# ─────────────────────────────────────────────────────────────────────────
# V5 — split the acquisition into a **value channel + uncertainty channel**
# (plan §8)
#
# Define the deterministic counterpart of acquisition A as the σ→0 limit:
#
#       A₀(x) = lim_{σ→0} A(x)        U(x) = A(x) − A₀(x)
#
# Shapley is linear in the value function (φ[v₁−v₂] = φ[v₁] − φ[v₂]), so
#
#       φ_k^A = φ_k^{A₀} + φ_k^U      ← **an identity, not an approximation**
#
# and each of the three channels satisfies efficiency. Standard BO can only
# say "this candidate was chosen because it is uncertain". It cannot say
# **which change is the uncertain part**.
#
# Key structure: in the target-oriented setting EXP, OPT and TS share the
# **same A₀ = −|μ−y*|**. Only tEI carries a clip, and if the clip is inactive
# on every coalition then A₀^tEI = e_best + A₀; Shapley cancels constants, so
# φ is **exactly equal**. Hence all differences between the four
# acquisitions live in the U channel:
#
#       φ^A − φ^{A'} = φ^{U_A} − φ^{U_{A'}}
#
# → "the choice of acquisition is entirely a choice of how to price
#   uncertainty" can be stated as a per-action identity.
#
# The sign is not fixed. In target-hitting, uncertainty is not always
# beneficial:
#   · U_EXP ≤ 0  **always** — by Jensen, E|Y−y*| ≥ |μ−y*|. Pure expected
#     utility is uncertainty-averse (the reverse of maximization BO).
#   · U_OPT > 0 ⟺ |μ−y*|/σ ≳ 0.1267 (α=0.1). Once already on target, even
#     the optimistic quantile loses. I.e. **far from target: opportunity;
#     on target: risk** — and our decomposition says which action is which.
#
# Cost is zero — μ already comes out of the single `moments(Z)` call, so A₀
# makes no additional surrogate call.
# ─────────────────────────────────────────────────────────────────────────

#: Point where the sign of U_OPT flips. Solved from setting
#: P(|Y−y*| ≤ |μ−y*|) = 1/2 − Φ(−2m/σ) equal to α — gate 21 checks this threshold.
def opt_sign_threshold(alpha: float = ALPHA) -> float:
    """Threshold t* on |μ−y*|/σ. U_OPT > 0 iff m/σ > t*."""
    return -0.5 * norm.ppf(0.5 - alpha)


def acq_plugin(name: str, mu: np.ndarray, y_star: float,
               e_best: float) -> np.ndarray:
    """A₀ = lim_{σ→0} A — determined by **μ alone**."""
    base = -np.abs(mu - y_star)
    if name in ("EXP", "OPT", "TS", "CB"):
        return base
    if name == "tEI":
        return np.maximum(0.0, e_best + base)
    raise ValueError(f"unknown acquisition: {name}")


def acq_split(surr, Z: np.ndarray, y_star: float, e_best: float,
              name: str, crn_key: str, alpha: float = ALPHA):
    """Return `(A, A₀, U)` together from **a single moments call**.

    All three plug directly into the `utility_fn` slot of `attribute_batch`,
    and the three decompositions lock together exactly as φ^A = φ^{A₀} + φ^U
    (gate 18).
    """
    mu, sd = _moments(surr, Z)
    a = _from_moments(mu, sd, Z, y_star, e_best, name, crn_key, alpha)
    a0 = acq_plugin(name, mu, y_star, e_best)
    return a, a0, a - a0


def tei_clip_active(mu: np.ndarray, y_star: float, e_best: float) -> float:
    """Fraction of rows where the clip is active in tEI's A₀.

    If 0, then φ^{A₀,tEI} == φ^{A₀,EXP} holds **exactly** over that coalition
    set (a constant shift). This measures the scope of the shared-baseline
    claim.
    """
    return float(np.mean(e_best - np.abs(mu - y_star) <= 0.0))
