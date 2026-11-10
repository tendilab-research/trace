"""Consistency gates 1–5 (plan §5). CPU only — runs without a GPU.

What these gates verify is that the action decomposition **never leaves the
coordinate system**. Shapley requires the utility of every subset, so if even
one subset steps outside the simplex, the surrogate is fed a physically
impossible recipe and the entire attribution loses meaning. This never shows
up at aggregation time.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracebo.actions import apply, decompose, renorm
from tracebo.data import load
from tracebo.meta import DATASETS
from tracebo.policies import pct_rank
from tracebo.shapley import EPS, attribute_batch, coherence, exact_shapley

DATASET_NAMES = list(DATASETS)
N_PAIRS = 400            # (ref, cand) pairs to check per dataset
SEED = 12345


def _pairs(pool, n_pairs=N_PAIRS, seed=SEED):
    rng = np.random.Generator(np.random.PCG64(seed))
    i = rng.integers(0, pool.n, n_pairs)
    j = rng.integers(0, pool.n, n_pairs)
    return [(pool.X[a], pool.X[b]) for a, b in zip(i, j) if a != b]


@pytest.fixture(scope="module", params=DATASET_NAMES)
def pool(request):
    return load(request.param)


# ── gate 1 ────────────────────────────────────────────────────────────────
def test_gate1_reconstruction(pool):
    """Applying all actions yields exactly the candidate (composition compared after renorm)."""
    m = pool.meta
    for ref, cand in _pairs(pool):
        acts = decompose(ref, cand, m)
        got = apply(ref, acts, m)
        want = cand.copy()
        if m.n_comp:
            want[m.comp_idx] = renorm(want[m.comp_idx], m.comp_total)
        assert np.allclose(got, want, atol=1e-9), \
            f"{m.name}: reconstruction failed, max error {np.abs(got - want).max():.3g}"


# ── gate 2 ────────────────────────────────────────────────────────────────
def test_gate2_simplex_preserved_on_every_subset(pool):
    """Composition sum preserved + non-negativity on **every subset**.

    Implementations exist that pass when only the full application is checked
    but break on subsets. Shapley uses the subsets, so they must be caught here.
    """
    m = pool.meta
    if not m.n_comp:
        pytest.skip("no composition")
    for ref, cand in _pairs(pool, n_pairs=60):
        acts = decompose(ref, cand, m)
        if len(acts) > 12:
            continue                       # 2^12 = 4096, test time limit
        for mask in range(1 << len(acts)):
            z = apply(ref, [acts[b] for b in range(len(acts)) if mask >> b & 1], m)
            tot = z[m.comp_idx].sum()
            assert abs(tot - m.comp_total) < 1e-9, \
                f"{m.name}: composition sum {tot} != {m.comp_total} on subset {mask:b}"
            assert z[m.comp_idx].min() >= -1e-12, \
                f"{m.name}: negative component {z[m.comp_idx].min():.3g} on subset {mask:b}"


def test_gate2_order_independence(pool):
    """apply is a set operation — reordering must give the same result.

    This must hold so that Shapley need not enumerate permutations.
    """
    m = pool.meta
    rng = np.random.Generator(np.random.PCG64(7))
    for ref, cand in _pairs(pool, n_pairs=100):
        acts = decompose(ref, cand, m)
        if len(acts) < 2:
            continue
        perm = list(rng.permutation(len(acts)))
        assert np.allclose(apply(ref, acts, m),
                           apply(ref, [acts[p] for p in perm], m), atol=1e-12)


# ── gate 3 ────────────────────────────────────────────────────────────────
def test_gate3_shapley_efficiency(pool):
    """Σφ_k == v(A) − v(∅)."""
    m = pool.meta
    rng = np.random.Generator(np.random.PCG64(3))
    ref = pool.X[0]
    cands = pool.X[rng.choice(pool.n, 24, replace=False)]

    # deterministic fake utility — tests Shapley itself without a surrogate
    def utility(Z):
        return np.sin(Z.sum(axis=1) * 3.0) + 0.5 * np.cos(Z[:, 0] * 7.0)

    att = attribute_batch(ref, cands, m, utility)      # internal assert already checks this
    for phi, du in zip(att.phi, att.delta_util):
        assert abs(phi.sum() - du) < 1e-8 * max(1.0, abs(du))


# ── gate 4 ────────────────────────────────────────────────────────────────
def test_gate4_dummy():
    """An action with no effect on the utility has φ = 0."""
    m = 4
    v = np.zeros(1 << m)
    for mask in range(1 << m):
        # only action 3 never appears in v → dummy
        v[mask] = sum((mask >> k & 1) * (k + 1) for k in range(3))
    phi = exact_shapley(v, m)
    assert abs(phi[3]) < 1e-12, f"dummy action has φ={phi[3]}"
    assert np.allclose(phi[:3], [1.0, 2.0, 3.0])       # additive utility → φ = individual contribution


def test_gate4_symmetry():
    """Two symmetric actions receive the same φ."""
    m = 3
    v = np.array([bin(mask).count("1") ** 2 for mask in range(1 << m)], dtype=float)
    phi = exact_shapley(v, m)
    assert np.allclose(phi, phi[0])                    # fully symmetric utility


# ── gate 5 ────────────────────────────────────────────────────────────────
def test_gate5_coherence_domain(pool):
    m = pool.meta
    rng = np.random.Generator(np.random.PCG64(5))
    ref = pool.X[0]
    cands = pool.X[rng.choice(pool.n, 32, replace=False)]

    def utility(Z):
        return rng.standard_normal(len(Z))             # worst case: pure noise

    att = attribute_batch(ref, cands, m, utility)
    assert np.all((att.coherence >= 0.0) & (att.coherence <= 1.0))
    assert not np.isnan(att.coherence).any()


def test_gate5_zero_contribution_is_half():
    """Exactly 0.5 when P+N ≈ 0. Setting 0/0 to 1 would give a 'no-contribution candidate' the top score."""
    assert coherence(np.zeros(5)) == 0.5
    assert coherence(np.array([EPS / 10, -EPS / 10])) == 0.5
    assert coherence(np.array([1.0, 2.0])) == pytest.approx(1.0, abs=1e-9)
    assert coherence(np.array([-1.0, -2.0])) == pytest.approx(0.0, abs=1e-9)


# ── gate 6: rank-product ──────────────────────────────────────────────────
def test_gate6_pct_rank_domain():
    """`q ∈ (0,1)`, zero NaN. A 0 makes the product 0 and wipes out the q_A
    information entirely, so the open-boundary `(r+0.5)/n` is a requirement."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 17, 256):
        for v in (rng.random(n), np.zeros(n), np.arange(n, dtype=float),
                  rng.integers(0, 3, n).astype(float)):
            q = pct_rank(v)
            assert q.shape == (n,)
            assert np.isfinite(q).all(), "NaN/inf produced"
            assert (q > 0).all() and (q < 1).all(), f"domain violation: {q.min()},{q.max()}"


def test_gate6_ties_get_average_rank():
    """Ties get the average rank. Breaking ties by order leaks the pool index
    order into the score, making determinism depend on data array order."""
    q = pct_rank(np.array([1.0, 1.0, 1.0, 2.0]))
    assert np.allclose(q[:3], q[0]), "tied values got different ranks"
    assert q[3] > q[0]
    # average rank: the three ties have rank (0+1+2)/3 = 1 → (1+0.5)/4
    assert np.isclose(q[0], 1.5 / 4)


def test_gate6_rank_product_needs_both_factors():
    """s = q_A·q_X wins only when **both** are high — that is why the product is used.

    A weighted sum `λq_A + (1−λ)q_X` would let one side near 0 be compensated
    by the other, and λ would need tuning. The product has no parameter.
    """
    qA = np.array([0.9, 0.5, 0.1])
    qX = np.array([0.1, 0.5, 0.9])
    s = qA * qX
    # the candidate with both moderate (1) must beat the two with only one side high (0, 2)
    assert s[1] > s[0] and s[1] > s[2]
    # raising one factor must raise that candidate's score (monotone in each factor)
    qX2 = qX.copy(); qX2[0] *= 2.0
    assert (qA * qX2)[0] > s[0]
    # with a weighted sum candidate 0 would have tied — make the difference explicit
    assert (0.5 * qA + 0.5 * qX)[0] == (0.5 * qA + 0.5 * qX)[1]


# ─────────────────────────────────────────────────────────────────────────
# gates 16·17 — V4 (acquisition decomposition). The entire claim of this design rests on two properties.
#   16  Σφ_k = acq(x) − acq(ref)        the explanation decomposes the decision **exactly**
#   17  acq(z) is batch-invariant        value at selection time == value at explanation time
# If 17 breaks, 16 degrades to an approximation — and the "zero loss" claim is gone.
# ─────────────────────────────────────────────────────────────────────────

def _toy_state(seed=0, n=24, ds="crossed_barrel"):
    import numpy as np
    from tracebo.data import load
    from tracebo.runner import _make_surrogate
    pool = load(ds)
    rng = np.random.default_rng(seed)
    obs = rng.choice(pool.n, n, replace=False)
    surr = _make_surrogate("GP", seed).fit(pool.X[obs], pool.y[obs])
    y_star = float(np.quantile(pool.y, 0.9))
    e_best = float(np.abs(pool.y[obs] - y_star).min())
    return pool, surr, y_star, e_best, obs


@pytest.mark.parametrize("name", ["EXP", "OPT", "tEI", "TS"])
@pytest.mark.parametrize("ds", DATASET_NAMES)
def test_gate16_acq_shapley_efficiency(name, ds):
    """Σφ_k == acq(x) − acq(ref). The entire basis of V4.

    Run on **all datasets**. Initially only crossed_barrel was checked, which
    missed TS breaking at 7.5e-3 on the two oer datasets (the coordinate hash
    crossing the rounding boundary of the reconstruction error). A gate is only
    meaningful where it is most fragile.
    """
    import numpy as np
    from tracebo.acquisitions import acq_values
    from tracebo.shapley import attribute_batch
    pool, surr, y_star, e_best, obs = _toy_state(ds=ds)
    ref = pool.X[obs[0]]
    cands = pool.X[[i for i in range(pool.n) if i not in set(obs)][:8]]

    def fn(Z):
        return acq_values(surr, Z, y_star, e_best, name, "gate16")

    att = attribute_batch(ref, cands, pool.meta, fn)
    a_ref = float(fn(ref[None, :])[0])
    a_cand = fn(cands)
    for k in range(len(cands)):
        lhs = float(att.phi[k].sum())
        rhs = float(a_cand[k] - a_ref)
        assert abs(lhs - rhs) < 1e-8 * max(1.0, abs(rhs)), \
            f"{name} candidate {k}: Σφ={lhs:.8g} != acq(x)−acq(ref)={rhs:.8g}"


@pytest.mark.parametrize("name", ["EXP", "OPT", "tEI", "TS"])
def test_gate17_acq_batch_invariance(name):
    """The same point must give the same value regardless of batch composition or order.

    The Monte Carlo path in `acquisitions()` does **not** have this property —
    which is why V4 uses `acq_values`. TS secures it via a coordinate hash
    (`_pointwise_normal`).

    The tolerance is relative to **the scale of the acquisition itself**. It must
    not be measured as pointwise relative error — the residual error comes from
    GP `moments`, not the acquisition. Even for the same point, a different batch
    shape changes the operation order of Cholesky/solve, μ jitters by ~4e-15, and
    tEI amplifies that in the tail (where values are 1e-5 of the scale). What
    matters is whether the identity `Σφ = acq(x) − acq(ref)` holds at the
    acquisition scale, and that is the criterion used here.
    """
    import numpy as np
    from tracebo.acquisitions import acq_values
    pool, surr, y_star, e_best, obs = _toy_state()
    Z = pool.X[:40]

    def fn(A):
        return acq_values(surr, A, y_star, e_best, name, "gate17")

    full = fn(Z)
    tol = 1e-11 * max(1e-30, float(np.abs(full).max()))
    # sub-batch
    assert np.allclose(fn(Z[:7]), full[:7], atol=tol, rtol=0)
    # reversed order
    assert np.allclose(fn(Z[::-1]), full[::-1], atol=tol, rtol=0)
    # single row
    for i in (0, 13, 39):
        assert np.allclose(fn(Z[i][None, :]), full[i], atol=tol, rtol=0)


def test_gate17b_sampled_acq_is_not_invariant():
    """Contrast — the existing Monte Carlo path is **not** batch-invariant.

    If this check fails, the reason for building `acq_values` separately is
    gone and the design rationale must be re-examined.
    """
    import numpy as np
    from tracebo.acquisitions import acquisitions, crn
    pool, surr, y_star, e_best, obs = _toy_state()
    Z = pool.X[:40]
    a = acquisitions(surr, Z, y_star, e_best, crn(0, 1, "x"))["EXP"]
    b = acquisitions(surr, Z[:7], y_star, e_best, crn(0, 1, "x"))["EXP"]
    assert not np.allclose(a[:7], b, rtol=1e-12)


def test_gate16b_selection_is_pure_acquisition():
    """V4 selection must be **exactly identical** to the baseline argmax (zero performance loss)."""
    import numpy as np
    from tracebo.acquisitions import acq_values
    from tracebo.policies import select_acq_explain, select_base
    pool, surr, y_star, e_best, obs = _toy_state()
    unobs = np.array([i for i in range(pool.n) if i not in set(obs)])
    acq = acq_values(surr, pool.X[unobs], y_star, e_best, "tEI", "g16b")
    ref = pool.X[obs[0]]

    def fn(Z):
        return acq_values(surr, Z, y_star, e_best, "tEI", "g16b")

    v4 = select_acq_explain(acq, unobs, ref, pool.X, pool.meta, fn)
    assert v4.idx == select_base(acq, unobs).idx
    assert v4.K_used == len(unobs)          # no K = the whole pool


# ─────────────────────────────────────────────────────────────────────────
# gates 18–22 — V5 (value channel / uncertainty channel decomposition).
#   18  φ^A = φ^{A₀} + φ^U               Shapley linearity, an identity
#   19  efficiency for each of the three channels
#   20  A → A₀ as σ→0 (rate O(σ))        validity of the A₀ definition
#   21  sign facts about U                in target matching, uncertainty cuts both ways
#   22  EXP·OPT·TS **share** φ^{A₀}       basis of the "all differences are in the U channel" claim
# ─────────────────────────────────────────────────────────────────────────

ACQ4 = ["EXP", "OPT", "tEI", "TS"]


def _v5_phis(name, e_best_override=None):
    """Return (φ^A, φ^{A₀}, φ^U, A, A₀, rows) for one candidate."""
    import numpy as np
    from tracebo.acquisitions import acq_split
    from tracebo.shapley import coalition_design, exact_shapley
    pool, surr, y_star, e_best, obs = _toy_state()
    if e_best_override is not None:
        e_best = e_best_override
    ref = pool.X[obs[0]]
    cand = pool.X[[i for i in range(pool.n) if i not in set(obs)][3]]
    Z, rows, acts = coalition_design(ref, cand[None, :], pool.meta)
    a, a0, u = acq_split(surr, Z, y_star, e_best, name, "gate18")
    r, m = rows[0], len(acts[0])
    assert m >= 2, "need a candidate with a non-trivial decomposition"
    return (exact_shapley(a[r], m), exact_shapley(a0[r], m),
            exact_shapley(u[r], m), a[r], a0[r], m)


@pytest.mark.parametrize("name", ACQ4)
def test_gate18_channel_linearity(name):
    """φ^A − φ^{A₀} == φ^U. Not an approximation but an identity from Shapley linearity."""
    import numpy as np
    pa, p0, pu, a, a0, m = _v5_phis(name)
    scale = max(1.0, float(np.abs(pa).max()))
    assert np.allclose(pa - p0, pu, atol=1e-11 * scale, rtol=0), \
        f"{name}: channel decomposition mismatch max={np.abs(pa - p0 - pu).max():.3e}"


@pytest.mark.parametrize("name", ACQ4)
def test_gate19_three_channel_efficiency(name):
    """Each of the three channels satisfies Σφ = (candidate value − reference value)."""
    import numpy as np
    pa, p0, pu, a, a0, m = _v5_phis(name)
    u = a - a0
    for phi, v, lab in ((pa, a, "A"), (p0, a0, "A0"), (pu, u, "U")):
        rhs = float(v[-1] - v[0])
        assert abs(float(phi.sum()) - rhs) < 1e-8 * max(1.0, abs(rhs)), \
            f"{name}/{lab}: Σφ={phi.sum():.8g} != {rhs:.8g}"


@pytest.mark.parametrize("name", ACQ4)
def test_gate20_plugin_is_the_sigma_zero_limit(name):
    """A → A₀ as σ→0. Confirm the O(σ) rate at two scales.

    With tEI as a layered integral this gate **stalls** at 6e-3 — the reason
    for switching to the closed form.
    """
    import numpy as np
    from tracebo.acquisitions import _from_moments, acq_plugin
    rng = np.random.default_rng(7)
    y_star, e_best = 1.0, 0.5
    mu = y_star + rng.normal(0.0, 0.6, 300)
    Z = mu[:, None]
    for s in (1e-3, 1e-6):
        sd = np.full_like(mu, s)
        a = _from_moments(mu, sd, Z, y_star, e_best, name, "g20", 0.1)
        gap = float(np.abs(a - acq_plugin(name, mu, y_star, e_best)).max())
        assert gap < 20.0 * s, f"{name}: |A−A₀|={gap:.3e} at σ={s:.0e}"


def test_gate21_uncertainty_sign():
    """In target matching, uncertainty is not unilaterally beneficial.

    · U_EXP ≤ 0 **always** — Jensen: E|Y−y*| ≥ |μ−y*|.
    · U_OPT is positive only when |μ−y*|/σ exceeds the threshold t*. I.e. when
      already on target, even the optimistic quantile is a loss. This is where
      it flips relative to maximization BO.
    """
    import numpy as np
    from tracebo.acquisitions import (_from_moments, acq_plugin,
                                      opt_sign_threshold)
    rng = np.random.default_rng(11)
    y_star, e_best = 0.0, 1.0
    sd = np.full(4000, 0.3)
    mu = rng.normal(0.0, 1.0, 4000)
    Z = mu[:, None]
    base = acq_plugin("EXP", mu, y_star, e_best)

    u_exp = _from_moments(mu, sd, Z, y_star, e_best, "EXP", "g21", 0.1) - base
    assert u_exp.max() <= 1e-12, f"U_EXP is positive: {u_exp.max():.3e}"

    u_opt = _from_moments(mu, sd, Z, y_star, e_best, "OPT", "g21", 0.1) - base
    r = np.abs(mu - y_star) / sd
    t_star = opt_sign_threshold(0.1)
    far, near = r > t_star * 1.5, r < t_star * 0.5
    assert (u_opt[far] > 0).all(), "far from target but optimism bonus is negative"
    assert (u_opt[near] < 0).all(), "on target but optimism bonus is positive"


def test_gate22_shared_plugin_baseline():
    """EXP·OPT·TS have **identical** φ^{A₀}, and tEI matches too when the clip is inactive.

    This is the basis of the claim "all differences between the four acquisitions are in the U channel":
        φ^A − φ^{A'} = φ^{U_A} − φ^{U_{A'}}
    """
    import numpy as np
    from tracebo.acquisitions import acq_split, tei_clip_active
    from tracebo.shapley import coalition_design, exact_shapley
    pool, surr, y_star, e_best, obs = _toy_state()
    ref = pool.X[obs[0]]
    cand = pool.X[[i for i in range(pool.n) if i not in set(obs)][3]]
    Z, rows, acts = coalition_design(ref, cand[None, :], pool.meta)
    r, m = rows[0], len(acts[0])
    mu, _ = surr.moments(Z)
    big_e = 10.0 * float(np.abs(mu - y_star).max())     # disables the clip
    assert tei_clip_active(mu, y_star, big_e) == 0.0

    ref_phi = None
    for name in ACQ4:
        eb = big_e if name == "tEI" else e_best
        _, a0, _ = acq_split(surr, Z, y_star, eb, name, "gate22")
        phi0 = exact_shapley(a0[r], m)
        if ref_phi is None:
            ref_phi = phi0
        else:
            assert np.allclose(phi0, ref_phi, atol=1e-12, rtol=0), \
                f"{name}: φ^A₀ differs from the shared baseline"

    # contrast — once the clip engages they need no longer agree (scope of the claim)
    _, a0t, _ = acq_split(surr, Z, y_star, e_best, "tEI", "gate22")
    assert tei_clip_active(mu, y_star, e_best) > 0.0
