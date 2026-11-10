"""Consistency gates 7–10 (plan §5). CPU only.

Gate 9 matters most. An oracle leak **leaves no trace** at aggregation time —
TRACE just looks very good, and it is discovered after publication. Two layers
block it: a runtime guard (OracleGuard) and a static symbol check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tracebo.acquisitions import acquisitions, crn
from tracebo.data import load
from tracebo.guards import OracleGuard, OracleLeak
from tracebo.runner import Episode, run_episode
from tracebo.surrogates.gp import GPSurrogate
from tracebo.targets import initial_design, make_target

SRC = Path(__file__).resolve().parents[1] / "src" / "tracebo"

# modules on the policy path — must never touch unobserved y here
POLICY_MODULES = ["policies.py", "acquisitions.py", "shapley.py", "actions.py",
                  "surrogates/gp.py", "surrogates/rf.py", "surrogates/tabpfn.py",
                  "surrogates/base.py"]


# ── gate 7a: determinism (same backend) ─────────────────────────────────────
@pytest.mark.parametrize("variant", ["base", "coherence"])
def test_gate7a_determinism(variant):
    """Same seed run twice → trajectories must be bit-identical."""
    pool = load("autoam")
    tgt = make_target(pool, 0.5)
    ep = Episode("autoam", 0.5, "GP", "tEI", variant, n0=5, seed=3, budget=8, K=64)
    a = run_episode(ep, pool, tgt)
    b = run_episode(ep, pool, tgt)
    assert [r.selected_idx for r in a] == [r.selected_idx for r in b]
    assert [r.y_selected for r in a] == [r.y_selected for r in b]


# ── gate 8: paired design ─────────────────────────────────────────────────
def test_gate8_shared_initial_design():
    """**Every variant** with the same (ds, target, n0, seed) gets the same D₀.

    Putting the variant into the D₀ seed breaks this. Then method differences
    mix with initialization differences and the paired comparison is invalid.
    """
    pool = load("p3ht")
    for q in (0.2, 0.5, 0.8):
        tgt = make_target(pool, q)
        for seed in range(5):
            d0 = [initial_design(pool, tgt, 5, seed) for _ in range(3)]
            assert all(np.array_equal(d0[0], d) for d in d0[1:])


def test_gate8_d0_excludes_hits():
    """G13 — D₀ must never contain a hit candidate."""
    for name in ("p3ht", "oer_plate_3496", "autoam"):
        pool = load(name)
        for q in (0.2, 0.5, 0.8):
            tgt = make_target(pool, q)
            hits = set(tgt.hit_idx.tolist())
            for seed in range(20):
                assert not (set(initial_design(pool, tgt, 5, seed).tolist()) & hits)


# ── gate 9: oracle leakage ────────────────────────────────────────────────
def test_gate9_static_no_reveal_in_policy_modules():
    """The `reveal_for_metrics` symbol must not appear in policy modules (static check)."""
    offenders = []
    for rel in POLICY_MODULES:
        tree = ast.parse((SRC / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "reveal_for_metrics":
                offenders.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "reveal_for_metrics":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"ORACLE LEAK (static): {offenders}"


def test_gate9_runtime_guard_blocks_unobserved():
    g = OracleGuard(np.arange(10, dtype=float))
    g.observe(3); g.observe(7)
    assert np.array_equal(g.y_of([3, 7]), [3.0, 7.0])
    with pytest.raises(OracleLeak):
        g.y_of([3, 4])
    assert np.array_equal(g.unobserved(10), [0, 1, 2, 4, 5, 6, 8, 9])


def test_gate9_no_leak_during_episode():
    """Zero OracleLeak while running a real episode.

    `run_episode` uses only `guard.y_of`, so touching unobserved y raises
    immediately. Passing = no leak path was executed.
    """
    pool = load("agnp")
    tgt = make_target(pool, 0.5)
    for variant in ("base", "coherence", "opposite", "direct-utility"):
        ep = Episode("agnp", 0.5, "GP", "tEI", variant, n0=5, seed=0, budget=6, K=32)
        assert len(run_episode(ep, pool, tgt)) == 6


# ── gate 10: TS fairness ──────────────────────────────────────────────────
def test_gate10_crn_key_excludes_variant():
    """The variant must not enter the random key — the core of G10."""
    kw = dict(dataset="agnp", target_q=0.5, surrogate="GP",
              acquisition="TS", n0=5, seed=0)
    base = Episode(variant="base", **kw)
    trace = Episode(variant="coherence", **kw)
    assert base.crn_key == trace.crn_key, "variant leaked into the random key"
    assert base.shard_id != trace.shard_id, "shard_id must distinguish variants"


def test_gate10_ts_same_posterior_sample():
    """base-TS and TRACE-TS in the same state see **the same** posterior sample.

    Requirement of G10 (common random numbers). If the variant enters the
    random seed this breaks, and then we cannot rule out that the difference
    between the two methods came from randomness rather than reranking.
    """
    pool = load("crossed_barrel")
    tgt = make_target(pool, 0.5)
    d0 = initial_design(pool, tgt, 5, 0)
    surr = GPSurrogate(seed=0).fit(pool.X[d0], pool.y[d0])
    unobs = np.setdiff1d(np.arange(pool.n), d0)

    kw = dict(dataset="crossed_barrel", target_q=0.5, surrogate="GP",
              acquisition="TS", n0=5, seed=0)
    base = Episode(variant="base", **kw)
    trace = Episode(variant="coherence", **kw)

    a = acquisitions(surr, pool.X[unobs], tgt.y_star, 1.0, crn(0, 1, base.crn_key))
    b = acquisitions(surr, pool.X[unobs], tgt.y_star, 1.0, crn(0, 1, trace.crn_key))
    assert np.array_equal(a["TS"], b["TS"]), "base and TRACE see different TS samples"

    # different t must differ (reusing the same sample every iteration kills exploration)
    c = acquisitions(surr, pool.X[unobs], tgt.y_star, 1.0, crn(0, 2, base.crn_key))
    assert not np.array_equal(a["TS"], c["TS"])
