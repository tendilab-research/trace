"""Paper figure generation (plan §8 V5). Reads `results/v5_480.npz`.

Physical size is fixed first. The NeurIPS 2026 workshop text width is a
5.5in single column, and figures go to PDF **at final size** — no rescaling
in LaTeX. That way the pt specified here equals the pt on the page. Minimum
font size 6pt is respected (the most conservative journal minimum in the
figures skill).

Colors are the Wong/Okabe-Ito colorblind-safe palette.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

W_FULL, W_HALF = 5.5, 2.7          # inch — NeurIPS text width
OUT = Path("figures")
NPZ = Path("results/v5_480.npz")

# Wong/Okabe-Ito. 4 acquisitions + shared baseline (black)
C = {"EXP": "#E69F00", "OPT": "#56B4E9", "tEI": "#009E73", "TS": "#CC79A7",
     "base": "#000000", "grey": "#7f7f7f"}
ACQ4 = ("EXP", "OPT", "tEI", "TS")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 0.9,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def _short(l):
    """`ni_load→fe_load (+0.0123)` → `Ni$\\to$Fe`. Keep labels from eating the axis."""
    import re
    l = re.sub(r"\s*\([^)]*\)$", "", l).replace("_load", "").replace("_", " ")
    if "→" in l:
        a, b = l.split("→")
        return f"{a.strip().title()}$\\to${b.strip().title()}"
    return l.strip()


def _nrm(v):
    s = float(np.abs(v).sum())
    return v / s if s > 1e-30 else v


def _save(fig, name):
    """PDF for the page, PNG for visual checking (same physical size)."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)


def _panel(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def _boot(v, n=2000, seed=0):
    """95% percentile interval from episode resampling."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ──────────────────────────────────────────────────────────────────────────
def fig1(d):
    """★ Headline — one shared value channel, four different uncertainty channels.

    Same candidate, same action axis. Left: the φ^{A₀} that all four
    acquisitions share **verbatim**; right: the φ^U, all different. This one
    figure is the paper's claim.
    """
    # Pick from the 480 stored cases — can be changed without rerunning.
    # Requirements: an m where labels are readable, **comparable magnitude of
    # the two channels** (bal), and the four φ^U actually differing (otherwise
    # the figure does not show the claim).
    cs = [json.loads(str(x)) for x in d["cases"]]
    def _pick(c):
        if not 4 <= c["m"] <= 6:
            return -1.0
        u = np.array([_nrm(np.array(c["phiU"][n])) for n in ACQ4])
        spread = float(np.abs(u - u.mean(axis=0)).mean())
        return c["bal"] * spread
    c = max(cs, key=_pick)
    m = c["m"]
    lab = [_short(l) for l in c["labels"]]
    y = np.arange(m)[::-1]

    fig, (a0, a1) = plt.subplots(
        1, 2, figsize=(W_FULL, 0.26 * m + 1.35), sharey=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.7], wspace=0.08))

    # Normalize each channel by Σ|φ|. The four acquisitions' φ^U have
    # **different units** (tEI in improvement units, the rest in y units) —
    # raw values on one axis make the tEI bars invisible. This figure's claim
    # is *which action* gets credit, not the magnitude. Magnitude is Fig 2a's job.
    # φ^{A₀} is shared by **EXP·OPT·TS only** (mean/quantile/sample of the same
    # utility, i.e. location functionals, indistinguishable under a degenerate
    # posterior). tEI adds a clip on top, so once the clip binds it is a
    # **different function** — same structure as EI₀ ≠ UCB₀. Draw the three in
    # black and tEI separately. Drawing all four as one bar would be false.
    p0 = _nrm(np.array(c["phi0all"]["EXP"]))
    p0t = _nrm(np.array(c["phi0all"]["tEI"]))
    a0.barh(y + 0.16, p0, height=0.32, color=C["base"], edgecolor="none",
            label="EXP, OPT, TS")
    a0.barh(y - 0.16, p0t, height=0.32, facecolor="none",
            edgecolor=C["tEI"], lw=0.8, label="tEI")
    a0.axvline(0, color="k", lw=0.5)
    a0.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    a0.set_yticks(y); a0.set_yticklabels(lab)
    lim = 1.06 * max(np.abs(p0).max(), np.abs(p0t).max(),
                     max(np.abs(_nrm(np.array(c["phiU"][n]))).max()
                         for n in ACQ4))
    a0.set_xlim(-lim, lim)
    a0.set_xlabel(r"$\varphi^{A_0}_k\,/\,\sum_j|\varphi^{A_0}_j|$")
    a0.text(0.0, 1.19, "one shared value channel",
            transform=a0.transAxes, fontsize=6.5, va="bottom")
    _panel(a0, "a", dx=-0.62, dy=1.19)

    h = 0.68 / 4
    for i, n in enumerate(ACQ4):
        a1.barh(y + (1.5 - i) * h, _nrm(np.array(c["phiU"][n])), height=h,
                color=C[n], edgecolor="none", label=n)
    a1.axvline(0, color="k", lw=0.5)
    a1.set_xlim(-lim, lim)
    a1.set_xlabel(r"$\varphi^{U}_k\,/\,\sum_j|\varphi^{U}_j|$")
    a1.text(0.0, 1.19, "four different uncertainty channels",
            transform=a1.transAxes, fontsize=6.5, va="bottom")
    a1.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    _panel(a1, "b", dx=-0.045, dy=1.19)

    _save(fig, "fig1_channels")
    plt.close(fig)
    return f"fig1  {c['dataset']} q={c['q']} seed={c['seed']} t={c['t']} m={m}"


# ──────────────────────────────────────────────────────────────────────────
def fig2(d):
    """Two pieces of evidence that the decomposition is not vacuous — U is half, and the channels differ."""
    ep, cols = d["ep"], list(d["ep_cols"])
    traj = d["traj"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(W_FULL, 1.75),
                                 gridspec_kw=dict(wspace=0.32))

    ts = np.unique(traj[:, 0]).astype(int)
    mu = np.array([traj[traj[:, 0] == t, 1].mean() for t in ts])
    ci = np.array([_boot(traj[traj[:, 0] == t, 1]) for t in ts])
    a0.fill_between(ts, ci[:, 0], ci[:, 1], color=C["tEI"], alpha=0.22, lw=0)
    a0.plot(ts, mu, "o-", color=C["tEI"], ms=2.6, mew=0)
    a0.axhline(0.5, color=C["grey"], lw=0.5, ls=(0, (3, 2)))
    a0.set_xlabel("BO iteration $t$")
    a0.set_ylabel(r"uncertainty share")
    a0.set_xticks(ts[::2])
    a0.text(0.97, 0.06, f"{len(ep)} episodes", transform=a0.transAxes,
            ha="right", fontsize=6, color=C["grey"])
    _panel(a0, "a")

    keys = [("r_u_exp_tei", "tEI"), ("r_u_exp_opt", "OPT"),
            ("r_u_exp_ts", "TS")]
    data = [ep[:, cols.index(k)][np.isfinite(ep[:, cols.index(k)])]
            for k, _ in keys]
    v = a1.violinplot(data, positions=[1, 2, 3], widths=0.72,
                      showextrema=False, showmedians=True)
    for b, (_, n) in zip(v["bodies"], keys):
        b.set_facecolor(C[n]); b.set_alpha(0.55); b.set_edgecolor("none")
    v["cmedians"].set_color("k"); v["cmedians"].set_linewidth(0.8)
    a1.axhline(1.0, color=C["base"], lw=0.8)
    a1.text(3.42, 1.0, r"$\varphi^{A_0}$: $\rho\equiv 1$" "\n(shared, err 0)",
            fontsize=6, va="center", ha="right", color=C["base"])
    a1.axhline(0.0, color=C["grey"], lw=0.5, ls=(0, (3, 2)))
    a1.set_xticks([1, 2, 3])
    a1.set_xticklabels([r"$U_{\mathrm{EXP}}$ vs $U_{\mathrm{%s}}$" % n
                        for _, n in keys])
    a1.set_ylabel(r"rank correlation $\rho$")
    a1.set_ylim(-1.05, 1.28)
    _panel(a1, "b")

    _save(fig, "fig2_share_orthogonality")
    plt.close(fig)
    return (f"fig2  U share {traj[:,1].mean():.3f}  "
            f"t1 {mu[0]:.3f} → t{ts[-1]} {mu[-1]:.3f}")


# ──────────────────────────────────────────────────────────────────────────
def fig3(d):
    """In target-matching the sign of uncertainty flips — closed form + measured distribution."""
    from scipy.stats import norm
    alpha = 0.1
    t_star = -0.5 * norm.ppf(0.5 - alpha)
    r = np.logspace(-2, 1.1, 400)                 # |μ−y*|/σ

    # Normalize to σ=1. U/σ is a function of r only.
    u_exp = -(2 * norm.pdf(r) + r * (2 * norm.cdf(r) - 1)) + r
    lo, hi = np.zeros_like(r), r + 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        go = (norm.cdf(mid - r) - norm.cdf(-mid - r)) < alpha
        lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
    u_opt = -0.5 * (lo + hi) + r

    fig, ax = plt.subplots(figsize=(W_HALF, 1.75))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(1e-2, t_star, color=C["grey"], alpha=0.13, lw=0)
    ax.plot(r, u_exp, color=C["EXP"], label=r"$U_{\mathrm{EXP}}$")
    ax.plot(r, u_opt, color=C["OPT"], label=r"$U_{\mathrm{OPT}}$")
    ax.axvline(t_star, color=C["base"], lw=0.6, ls=(0, (2, 1.6)))
    ax.text(t_star * 0.88, 0.62, r"$t^\ast=0.127$", rotation=90,
            fontsize=6, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\mu-y^\ast|/\sigma$")
    ax.set_ylabel(r"$U/\sigma$")
    ax.set_xlim(1e-2, 12)
    ax.set_ylim(-1.0, 1.75)
    ax.legend(loc="upper left", handlelength=1.2, bbox_to_anchor=(0.02, 1.0))
    # Reading of this figure — the point where it flips relative to maximization BO
    ax.text(0.030, -0.90, "on target:\nliability", fontsize=6,
            color=C["grey"], ha="center", va="bottom")
    ax.text(3.2, -0.90, "far from target:\nopportunity", fontsize=6,
            color=C["grey"], ha="center", va="bottom")

    frac = float((d["traj"][:, 2] > t_star).mean())
    ax.text(0.99, 0.97, f"{frac*100:.1f}% of iterations lie here",
            transform=ax.transAxes, fontsize=6, color=C["grey"],
            ha="right", va="top")
    _save(fig, "fig3_sign_reversal")
    plt.close(fig)
    return f"fig3  t*={t_star:.4f}  far-region fraction {frac:.3f}"


def main():
    if not NPZ.exists():
        print(f"  {NPZ} missing — run uncert_split.py --out first", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ, allow_pickle=False)
    for f in (fig1, fig2, fig3):
        print("  " + f(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
