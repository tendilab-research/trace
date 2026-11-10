"""Additional figures — vector weighting (§4.6) and robustness (§4.3, Appendix).

Uses the same physical size and palette as `figures.py`. Both figures read
**only already-computed result files**: `actionw.npz`, `robust_n0.npz`,
`robust_surr.npz`, `robust_tabpfn.npz`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import C, OUT, W_FULL, _panel, _save                # noqa: E402

R = Path("results")
ACQ4 = ("EXP", "OPT", "tEI", "TS")


def _ld(name):
    d = np.load(R / f"{name}.npz", allow_pickle=True)
    df = pd.DataFrame(d["rows"], columns=[str(c) for c in d["cols"]])
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except (ValueError, TypeError):
            pass
    return df


def _boot_ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


# ══════════════════════════════════════════════════════════════════════════
def fig4():
    """Asymmetry is universal — and the size of the gain differs per acquisition.

    (a) Mirror-pair contrast. Two configs that are **the same two numbers in
        swapped order**, so there is no free parameter: `w=(0.25,1)` discounts
        uncertainty's *nudge* and trusts its *veto*, `w=(1,0.25)` is the
        reverse. All four negative = all four in the same direction.
    (b) One vector point on top of the scalar-sweep curve. Only TS has the
        vector drop **below** the scalar curve; the other three stay near the
        curve's endpoint (w=1, i.e. the standard acquisition). That is the
        honest claim of §4.6.
    """
    df = _ld("actionw")
    df["key"] = df.dataset + "|" + df.q.astype(str) + "|" + df.seed.astype(str)

    def cell(acq, cfg, w):
        return df[(df.acq == acq) & (df.cfg == cfg) & (df.w == w)] \
            .set_index("key").sort_index()

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(W_FULL, 1.85),
                                 gridspec_kw=dict(wspace=0.34, width_ratios=[1.0, 1.25]))

    y = np.arange(len(ACQ4))[::-1]
    for i, acq in enumerate(ACQ4):
        A = cell(acq, "add-sign", "0.25/1")["auc_err_n"]
        B = cell(acq, "add-sign", "1/0.25")["auc_err_n"]
        d = (A - B).to_numpy()
        lo, hi = _boot_ci(d)
        a0.barh(y[i], d.mean(), height=0.62, color=C[acq], edgecolor="none")
        a0.plot([lo, hi], [y[i], y[i]], color="k", lw=0.8, solid_capstyle="butt")
    a0.axvline(0, color="k", lw=0.5)
    a0.set_yticks(y); a0.set_yticklabels(list(ACQ4))
    a0.set_xlabel(r"$\Delta$ target-error AUC:  $w=(0.25,1)$ minus $(1,0.25)$")
    a0.set_title("trusting the veto beats trusting the nudge",
                 fontsize=6.5, pad=4)
    a0.text(0.03, -0.30, "480 episodes, paired Wilcoxon; bars are 95% bootstrap CI",
            transform=a0.transAxes, fontsize=6, color=C["grey"])
    _panel(a0, "a", dx=-0.30)

    ws = ["0", "0.25", "0.5", "0.75", "1"]
    xs = [float(w) for w in ws]
    for acq in ACQ4:
        m = [cell(acq, "add-scalar", w)["auc_err_n"].mean() for w in ws]
        a1.plot(xs, m, "-", color=C[acq], lw=0.9, marker="o", ms=2.2, mew=0,
                label=acq)
        v = cell(acq, "add-sign", "0.25/1")["auc_err_n"].mean()
        a1.plot([1.06 + 0.018 * ACQ4.index(acq)], [v], marker="*", ms=6.5,
                color=C[acq], mew=0, clip_on=False)
    a1.axvline(1.0, color=C["grey"], lw=0.5, ls=(0, (3, 2)))
    a1.set_xlim(-0.05, 1.20)
    a1.set_xticks(xs + [1.06])
    a1.set_xticklabels(ws + [r"$\bigstar$"])
    a1.set_xlabel(r"scalar weight $\lambda$   ($\lambda{=}1$ is the acquisition itself)")
    a1.set_ylabel("target-error AUC")
    a1.set_title(r"$\bigstar$ = sign-split vector $w=(0.25,1)$",
                 fontsize=6.5, pad=4)
    a1.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.06),
              handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    _panel(a1, "b", dx=-0.17)

    _save(fig, "fig4_weighting")
    plt.close(fig)
    return "fig4  weighting"


# ══════════════════════════════════════════════════════════════════════════
def fig5():
    """What is predicted does not change; only what is measured does.

    Perturb two axes (posterior **shape**, posterior **width**). The shared
    baseline error stays exactly 0 on both — that is the proposition's
    prediction — while the uncertainty channel's magnitude and sign ratio
    move substantially.
    """
    n0 = _ld("robust_n0")
    su = pd.concat([_ld("robust_surr"), _ld("robust_tabpfn")], ignore_index=True)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(W_FULL, 1.8),
                                 gridspec_kw=dict(wspace=0.58))

    g = n0.groupby("n0")
    xs = sorted(g.groups)
    sh = [g.get_group(k)["share_ch"].mean() for k in xs]
    ci = [_boot_ci(g.get_group(k)["share_ch"].to_numpy()) for k in xs]
    ci = np.array(ci)
    a0.fill_between(xs, ci[:, 0], ci[:, 1], color=C["tEI"], alpha=0.20, lw=0)
    a0.plot(xs, sh, "o-", color=C["tEI"], ms=2.6, mew=0, label="uncertainty share")
    a0.set_xscale("log")
    a0.set_xticks(xs); a0.set_xticklabels([f"{int(k)}" for k in xs])
    a0.minorticks_off()
    a0.set_ylim(0.0, 0.85)
    a0.set_xlabel(r"initial observations $n_0$")
    a0.set_ylabel("uncertainty share")
    b0 = a0.twinx()
    b0.plot(xs, [g.get_group(k)["auc_err_n"].mean() for k in xs], "s--",
            color=C["grey"], ms=2.2, mew=0, lw=0.8)
    b0.set_ylabel("target-error AUC", color=C["grey"], labelpad=1)
    b0.tick_params(axis="y", colors=C["grey"])
    b0.spines["right"].set_visible(True)
    b0.spines["right"].set_color(C["grey"])
    b0.set_ylim(0, 0.045)
    a0.text(0.5, 0.09, r"shared-baseline error $=0$ at every $n_0$",
            transform=a0.transAxes, ha="center", fontsize=6, color="k")
    a0.set_title("more data does not close the uncertainty channel",
                 fontsize=6.5, pad=4)
    _panel(a0, "a", dx=-0.24)

    names = ["GP", "RF", "TabPFN"]
    x = np.arange(3)
    sh = [su[su.surrogate == n]["share_ch"].mean() for n in names]
    er = np.array([_boot_ci(su[su.surrogate == n]["share_ch"].to_numpy())
                   for n in names])
    a1.bar(x, sh, width=0.55, color=[C["EXP"], C["OPT"], C["TS"]],
           edgecolor="none")
    for i in range(3):
        a1.plot([x[i], x[i]], er[i], color="k", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylim(0, 0.85)
    a1.set_ylabel("uncertainty share")
    for i, n in enumerate(names):
        s = su[su.surrogate == n]["sd_ref"].mean()
        a1.text(x[i], sh[i] + 0.055, rf"$\bar\sigma={s:.2f}$", ha="center",
                fontsize=6)
    a1.text(0.5, 0.09, r"shared-baseline error $=0$ for all three",
            transform=a1.transAxes, ha="center", fontsize=6, color="k")
    a1.set_title(r"posterior shape moves $\sigma$ 9$\times$, not the identity",
                 fontsize=6.5, pad=4)
    _panel(a1, "b", dx=-0.24)

    _save(fig, "fig5_robustness")
    plt.close(fig)
    return "fig5  robustness"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("  " + fig4())
    print("  " + fig5())
    return 0


if __name__ == "__main__":
    sys.exit(main())
