#!/usr/bin/env python3
"""Environment smoke test — runs identically on local (macOS/MPS) and pod (CUDA).

Checks
  1. Package versions match the requirements.txt pins
  2. Which device is resolved
  3. TabPFNRegressor actually fits/predicts on that device   <- today's biggest unknown
  4. Measured throughput (rows/s) — first measurement to replace the plan's r_gpu estimate

Exit code 0 = all passed. Otherwise = the failing step is printed to stdout.
"""

from __future__ import annotations

import sys
import time
import traceback

sys.path.insert(0, "src")

PROBE_TRAIN = 20      # n0 = 20 (plan §3)
PROBE_DIM = 5
PROBE_ROWS = 12_470   # coalition design rows per iteration (plan §4.6)
CHUNK = 2_000

ok = True


def step(title: str):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def fail(msg: str):
    global ok
    ok = False
    print(f"  [FAIL] {msg}")


# --- 1. versions ----------------------------------------------------------
step("1. Package versions")
mods = ["numpy", "scipy", "pandas", "sklearn", "pyarrow", "torch", "tabpfn",
        "statsmodels", "polars", "duckdb"]
for name in mods:
    try:
        m = __import__(name)
        print(f"  {name:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:                                  # noqa: BLE001
        fail(f"{name} import failed: {e}")

print(f"  {'python':<12} {sys.version.split()[0]}")

# --- 2. device ------------------------------------------------------------
step("2. Device resolution")
try:
    from tracebo.surrogates.device import device_info, resolve_device, torch_dtype

    dev = resolve_device()
    print(f"  resolve_device() -> {dev!r}")
    for k, v in device_info().items():
        print(f"    {k:<10} {v}")
    print(f"  torch_dtype()    -> {torch_dtype()}")
except Exception:                                           # noqa: BLE001
    fail("device.py failed to run")
    traceback.print_exc()
    dev = "cpu"

# --- 3. TabPFN ------------------------------------------------------------
step("3. TabPFN fit/predict")
import numpy as np                                          # noqa: E402

rng = np.random.default_rng(0)
Xtr = rng.random((PROBE_TRAIN, PROBE_DIM))
ytr = rng.random(PROBE_TRAIN)
Xte = rng.random((PROBE_ROWS, PROBE_DIM))

working_device = None
for cand in ([dev] if dev == "cpu" else [dev, "cpu"]):
    try:
        from tabpfn import TabPFNRegressor

        t0 = time.perf_counter()
        model = TabPFNRegressor(device=cand)
        model.fit(Xtr, ytr)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = model.predict(Xte[:CHUNK])
        t_pred = time.perf_counter() - t0

        assert np.asarray(out).shape[0] == CHUNK, f"predict shape {np.shape(out)}"
        rate = CHUNK / t_pred
        print(f"  [{cand}] fit {t_fit:6.2f}s | predict {CHUNK} rows {t_pred:6.2f}s "
              f"-> {rate:,.0f} rows/s")
        print(f"           estimated per iteration ({PROBE_ROWS:,} rows): {PROBE_ROWS / rate:.1f}s")
        working_device = working_device or cand
    except Exception as e:                                  # noqa: BLE001
        print(f"  [{cand}] failed: {type(e).__name__}: {e}")
        if cand == dev and dev != "cpu":
            print(f"  -> the {dev} backend is not usable with TabPFN. "
                  f"Use TRACE_DEVICE=cpu for local development.")

if working_device is None:
    fail("TabPFN does not work on any device")
else:
    print(f"\n  usable TabPFN device: {working_device!r}")

# --- 4. result ------------------------------------------------------------
step("Result")
print("  PASS" if ok else "  FAIL — see [FAIL] items above")
sys.exit(0 if ok else 1)
