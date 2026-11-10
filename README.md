# Why Did Bayesian Optimization Recommend This Recipe?

**Separating Value and Uncertainty for Each Recipe Change**

Minseong Kim, Junyoung Lee, Jiwoo Lee, Jinwon Yoon (TendiLab)

Code and experiment artifacts for the AI4Mat-NeurIPS 2026 workshop paper.

Bayesian optimization (BO) can recommend a material recipe for a target
property. This repository explains how each composition or process change from
the best observed recipe contributes to that recommendation's acquisition score.
The acquisition `A` is split into a deterministic-value component and an
uncertainty-induced residual,

    A₀(x) = lim_{σ→0} A(x)          U(x) = A(x) − A₀(x)

and, because Shapley values are linear, the per-change attribution splits
exactly the same way: `φ_k^A = φ_k^{A₀} + φ_k^U`. The repository contains the
implementation, the experiment scripts, the raw benchmark data, and the saved
result files behind the reported results.

## Layout

```
src/tracebo/        library: surrogates (GP / RF / TabPFN), acquisition functions,
                    action space, Shapley attribution, episode runner, sanity gates
scripts/            experiment entry points (see "Reproducing the paper")
analysis/           aggregation, bootstrap CIs, effect sizes
tests/              pytest sanity gates (identity checks, runtime gates)
data/raw/           benchmark datasets copied verbatim from upstream (see data/MANIFEST.md)
results/            saved .npz outputs that the paper's tables/figures are built from
figures/            output directory for scripts/figures*.py and scripts/table1.py
```

## Setup

Python 3.11 is required (`requires-python = ">=3.11,<3.12"`). All dependency
versions are pinned in `requirements.txt`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
export TRACE_DEVICE=cpu        # or cuda; see src/tracebo/surrogates/device.py
python scripts/verify_env.py   # smoke test: versions, device, TabPFN fit/predict
pytest                         # sanity gates
```

`scripts/setup_local.sh` (macOS, uses `uv`) and `scripts/setup_pod.sh`
(Linux + CUDA) automate the same steps. TabPFN weights are downloaded on first
use; set `TABPFN_MODEL_CACHE_DIR` to control where they are cached.

## Reproducing the paper

Every script reads only `data/raw/` and writes an `.npz` into `results/`
(defaults shown). The `results/` directory already contains the files used in
the paper, so figures and tables can be regenerated without re-running the
experiments.

| Paper element | Script | Output |
|---|---|---|
| Main replay: 480 episodes, GP, n₀=10 (§3, §4.2, Fig. 2) | `python scripts/uncert_split.py --out results/v5_480.npz` | `results/v5_480.npz` |
| Exactness checks and additive confidence-bound control (§4.1) | `python scripts/verify_claims.py`<br>`python scripts/table1.py --out figures/table1.tex` | stdout, `figures/table1.tex` |
| Family-level statistics: family-blocked bootstrap CIs and sign tests (Appendix B, Table 1) | `python scripts/family_stats.py` | `results/family_stats.{npz,md}` |
| n₀ and surrogate robustness (Appendix C, Figs. 3–4) | `python scripts/robustness.py --surrogates GP,RF,TabPFN --out results/robust_surr.npz`<br>`python scripts/robustness.py --n0-list 10,20,30,50,100,200,300 --out results/robust_n0.npz` | `results/robust_*.npz` |
| Synthetic controls (Appendix D) | `python scripts/explain_eval.py --out results/explain_eval.npz` | `results/explain_eval.npz` |
| Scalar weighting `A_λ = A₀ + λU` (§4.3) | `python scripts/lambda_sweep.py --out results/lambda.npz` | `results/lambda.npz` |
| Recipe-specific (sign-weighted) uncertainty weighting (§4.3, Appendix E, Fig. 5) | `python scripts/action_weight.py --out results/actionw.npz` | `results/actionw.npz` |
| Approximation gates (TabPFN ensemble size, top-K shortlist) | `python scripts/gate_1314.py` | stdout |
| Plots from the saved results | `python scripts/figures.py`, `python scripts/figures_v6.py` | `figures/*.pdf` |

The paper's figures were restyled for publication; the plotting scripts
regenerate the same quantities from the saved result files. The per-candidate
explanation timing in Appendix D was measured in a separate environment and is
not reproduced by these scripts.

Some code comments refer to section numbers of an earlier manuscript draft.

Run `python scripts/<name>.py --help` for options (`--seeds`, `--budget`,
`--n0`, `--datasets`, `--workers`, ...). GP and RF runs are CPU-only; TabPFN
runs are practical only on a GPU.

## Data

`data/raw/` contains 5 datasets from the Liang et al. benchmark
(`PV-Lab/Benchmarking`) and 8 datasets from Olympus
(`aspuru-guzik-group/olympus`), copied verbatim at pinned commits.
`data/MANIFEST.md` lists the source commit SHAs, per-file row counts, input
dimensions, target columns and checksums, and the commands used to reproduce
them. The datasets remain under their upstream licenses.

## Citation

```bibtex
@inproceedings{kim2026trace,
  title     = {Why Did Bayesian Optimization Recommend This Recipe? Separating Value and Uncertainty for Each Recipe Change},
  author    = {Kim, Minseong and Lee, Junyoung and Lee, Jiwoo and Yoon, Jinwon},
  booktitle = {AI for Accelerated Materials Design (AI4Mat) Workshop at NeurIPS 2026},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
