# TRACE — Data Manifest

Generated: 2026-08-29

Raw datasets used by the TRACE project, copied verbatim from two upstream repositories.
All counts below were computed directly from the copied files (see *Reproducing the counts*).

## Sources

| Repo | URL | Commit SHA (at clone) | Commit date |
|---|---|---|---|
| Liang benchmark (`PV-Lab/Benchmarking`) | https://github.com/PV-Lab/Benchmarking | `7585c517ad88e676c42c6bf24a8ad278e01ddb21` | 2021-11-18 |
| Olympus (`aspuru-guzik-group/olympus`) | https://github.com/aspuru-guzik-group/olympus | `440b6b58ebfcaa2391cff7e94b570fb4fda98d68` | 2023-08-13 |

Clone command used (shallow, into a temp dir; temp clones removed afterwards):

```bash
git clone --depth 1 https://github.com/PV-Lab/Benchmarking.git
git clone --depth 1 https://github.com/aspuru-guzik-group/olympus.git
```

## Directory layout

```
data/
├── raw/
│   ├── liang/          # 5 CSVs from Benchmarking/datasets/
│   └── olympus/        # 8 dataset dirs from olympus/src/olympus/datasets/
│                       #   (data.csv + config.json + description.txt only)
└── MANIFEST.md
```

## 1. Liang benchmark datasets (`data/raw/liang/`)

Source: `Benchmarking/datasets/<file>` @ `7585c517ad88`.
CSVs **have a header row**. Last column is the target `y`; all preceding columns are the input vector `X`.

| File | Rows (data) | Unique X rows | Input dim | Target column | sha256 (first 16) |
|---|---:|---:|---:|---|---|
| `data/raw/liang/AgNP_dataset.csv` | 3295 | 164 | 5 | `loss` | `d25ec6fbb178bbb1` |
| `data/raw/liang/P3HT_dataset.csv` | 233 | 178 | 5 | `Conductivity (measured) (S/cm)` | `db33eeb63e3f09eb` |
| `data/raw/liang/Perovskite_dataset.csv` | 139 | 94 | 3 | `Instability index` | `9e9c5c2c698902ae` |
| `data/raw/liang/AutoAM_dataset.csv` | 100 | 100 | 4 | `Score` | `01ec270fcaae908f` |
| `data/raw/liang/Crossed barrel_dataset.csv` | 1800 | 600 | 4 | `toughness` | `2c01f875f3c210e9` |

Input columns:

- **AgNP_dataset.csv** — `QAgNO3(%)`, `Qpva(%)`, `Qtsc(%)`, `Qseed(%)`, `Qtot(uL/min)`
- **P3HT_dataset.csv** — `P3HT content (%)`, `D1 content (%)`, `D2 content (%)`, `D6 content (%)`, `D8 content (%)`
- **Perovskite_dataset.csv** — `CsPbI`, `FAPbI`, `MAPbI`
- **AutoAM_dataset.csv** — `Prime Delay`, `Print Speed`, `X Offset Correction`, `Y Offset Correction`
- **Crossed barrel_dataset.csv** — `n`, `theta`, `r`, `t`

Notes:

- `Perovskite_dataset.csv` has a UTF-8 BOM on the first header cell; read it with `encoding='utf-8-sig'`.
- `AgNP`, `Crossed barrel` and `Perovskite` contain **replicate measurements**: the same input vector appears
  multiple times with different `y`. `AgNP` = 3295 rows / 164 unique X (~20x replicates),
  `Crossed barrel` = 1800 / 600 (exactly 3x), `Perovskite` = 139 / 94, `P3HT` = 233 / 178.
  `AutoAM` has no duplicate inputs (100/100).

## 2. Olympus datasets (`data/raw/olympus/`)

Source: `olympus/src/olympus/datasets/<dir>` @ `440b6b58ebfc`.
Each directory holds `data.csv`, `config.json`, `description.txt` (notebooks/images were not copied).

`data.csv` has **no header row**. Column order is `config.json`'s `parameters` (in order), then `measurements`.

| Dataset | Rows | Unique X rows | Input dim | Target column | `constraints.parameters` | Goal | data.csv sha256 (first 16) |
|---|---:|---:|---:|---|---|---|---|
| `data/raw/olympus/dataset_oer_plate_3496/` | 2121 | 2121 | 6 | `overpotential` | `simplex` | minimize | `3c70049ccfdd11bc` |
| `data/raw/olympus/dataset_oer_plate_4098/` | 2121 | 2121 | 6 | `overpotential` | `simplex` | minimize | `a3e4b4b781e3a04f` |
| `data/raw/olympus/dataset_oer_plate_3851/` | 2119 | 2119 | 6 | `overpotential` | `simplex` | minimize | `e2212be9cc5c866f` |
| `data/raw/olympus/dataset_oer_plate_3860/` | 2120 | 2120 | 6 | `overpotential` | `simplex` | minimize | `834e2832818900e5` |
| `data/raw/olympus/dataset_hplc/` | 1386 | 1007 | 6 | `peak_area` | `none` | maximize | `9c94222798229c13` |
| `data/raw/olympus/dataset_photo_pce10/` | 1040 | 1020 | 4 | `degradation` | `simplex` | minimize | `b41c72b8e38e73a5` |
| `data/raw/olympus/dataset_p3ht/` | 178 | 178 | 5 | `conductivity` | `none` | maximize | `be832eb97e1e18f4` |
| `data/raw/olympus/dataset_agnp/` | 164 | 164 | 5 | `spectrum_score` | `none` | maximize | `a6716a575a9890fc` |

Input columns (in `data.csv` column order):

- **dataset_oer_plate_3496** — `ni_load`, `fe_load`, `co_load`, `mn_load`, `ce_load`, `la_load`
- **dataset_oer_plate_4098** — `sn_load`, `sb_load`, `co_load`, `ca_load`, `ni_load`, `mn_load`
- **dataset_oer_plate_3851** — `ni_load`, `fe_load`, `co_load`, `ta_load`, `mn_load`, `cu_load`
- **dataset_oer_plate_3860** — `sn_load`, `fe_load`, `co_load`, `ta_load`, `mn_load`, `cu_load`
- **dataset_hplc** — `sample_loop`, `additional_volume`, `tubing_volume`, `sample_flow`, `push_speed`, `wait_time`
- **dataset_photo_pce10** — `mat_1`, `mat_2`, `mat_3`, `mat_4`
- **dataset_p3ht** — `p3ht_content`, `d1_content`, `d2_content`, `d6_content`, `d8_content`
- **dataset_agnp** — `q_agno3`, `q_pva`, `q_tsc`, `q_seed`, `q_tot`

Notes:

- **Simplex constraint** (`constraints.parameters == "simplex"`) applies to the four `oer_plate_*` sets and
  `photo_pce10`: the input components are fractional compositions that sum to 1, so the feasible region is a
  (d-1)-simplex, not a box. `hplc`, `p3ht` and `agnp` are unconstrained boxes (`"none"`).
- `description.txt` for `dataset_oer_plate_3851` and `dataset_oer_plate_3860` both state "Number of Samples 2121",
  but the shipped `data.csv` files actually contain **2119** and **2120** rows respectively. The row counts in the
  table above are the true file contents.
- Only `hplc` (1386 / 1007) and `photo_pce10` (1040 / 1020) contain duplicate input vectors; every other
  Olympus set here is already deduplicated.
- Purpose: `oer_plate_3496` / `oer_plate_4098` are the primary simplex targets, `oer_plate_3851` / `oer_plate_3860`
  are held in reserve; `p3ht` and `agnp` exist to cross-check against the Liang versions of the same experiments.

## 3. Liang / Olympus cross-check

The Olympus `p3ht` and `agnp` sets are the replicate-averaged form of the corresponding Liang CSVs — their row
counts match the Liang unique-X counts exactly, which confirms the dedup logic used above:

| Experiment | Liang rows | Liang unique X | Olympus rows | Match |
|---|---:|---:|---:|:--:|
| AgNP | 3295 | 164 | 164 | yes |
| P3HT | 233 | 178 | 178 | yes |

(The `y` values differ: Liang stores each individual replicate measurement, Olympus stores one aggregated value
per composition — e.g. Liang AgNP row 1 `loss=0.57959227` vs Olympus `spectrum_score=0.585862380576923`.)

## 4. Reproducing the counts

Row counts are data rows only (header excluded for Liang, no header exists for Olympus; blank lines skipped).
Unique-X counts drop the last column, parse every remaining cell as a float, and count distinct tuples:

```python
import csv

def counts(path, has_header, encoding='utf-8'):
    with open(path, newline='', encoding=encoding) as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if has_header:
        rows = rows[1:]
    uniq = {tuple(float(c) for c in r[:-1]) for r in rows}
    return len(rows), len(uniq)
```

Olympus `config.json` files use tabs and may contain `//` comments; strip comments
(`re.sub(r'(?m)//.*$', '', text)`) before `json.loads`.
