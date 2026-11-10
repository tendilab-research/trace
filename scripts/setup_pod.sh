#!/usr/bin/env bash
# Run inside a GPU pod. No custom Docker image is built.
#
# Rationale: the reason for baking in the checkpoint (G12) was "many workers
# hitting HuggingFace at once get rate-limited", but there is now a single pod.
# Cross-building a linux/amd64 image on Apple Silicon, on the other hand, puts
# 20–40 minutes of emulation on the critical path. The provider's stock PyTorch
# template + this script is strictly better.
#
# A separate python 3.11 venv is created. Partly because the template default is
# 3.12 and pyproject requires >=3.11,<3.12, but mainly to **match package
# versions exactly with local**. Then the residual difference in gate 7b
# (cross-backend agreement) narrows to CUDA kernel nondeterminism alone. The pod
# link is 18MB/s, so reinstalling torch (2.5GB) takes 3 minutes — the opposite
# of the local link (66KB/s).
#
# Usage (inside the pod):
#   cd /workspace/trace && bash scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> checking GPU"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
  echo "nvidia-smi failed — this pod has no GPU. Aborting."; exit 1; }

echo "==> uv venv (python 3.11)"
uv venv --python 3.11 .venv

echo "==> installing dependencies (overrides the template's torch with the pin)"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps

echo "==> warming up TabPFN checkpoint (one-time HuggingFace download)"
# Keep the cache on the **volume** (/workspace). On the overlay root it is lost when the pod is recreated.
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-/workspace/.tabpfn}"
mkdir -p "$TABPFN_MODEL_CACHE_DIR"
.venv/bin/python - <<'PY'
import numpy as np
from tracebo.surrogates.tabpfn import _make_regressor
m = _make_regressor("cpu", 1, 0)             # CPU — only the checkpoint download is needed
m.fit(np.random.rand(20, 4), np.random.rand(20))
m.predict(np.random.rand(5, 4))
print("checkpoint cached")
PY

echo "==> pinning threads (so N workers do not fight over BLAS)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "==> smoke test"
TRACE_DEVICE=cuda .venv/bin/python scripts/verify_env.py

cat <<'MSG'

Next:
  cd /workspace/trace && source .venv/bin/activate
  export TRACE_DEVICE=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export TABPFN_MODEL_CACHE_DIR=/workspace/.tabpfn
MSG
