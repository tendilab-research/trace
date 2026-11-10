#!/usr/bin/env bash
# Local (macOS/arm64) native dev environment. No Docker.
#   - Docker Desktop cannot pass Metal (MPS) through to a container.
#   - The CUDA image for the pod is linux/amd64 and does not run usefully here.
# So local = venv, pod = container. Version pins are shared via requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=3.11

echo "==> uv venv (python ${PY})"
uv venv --python "${PY}" .venv

echo "==> installing dependencies"
# On macOS/arm64 tabpfn pulls in mlx + mlx-metal (40MB). Useless while local runs
# on CPU, and on a slow link this alone takes 20+ minutes. The linux pod never
# fetches it anyway (sys_platform=="darwin" marker), so there is no divergence.
grep -v '^tabpfn==' requirements.txt > /tmp/trace-req-nomlx.txt
uv pip install --python .venv/bin/python -r /tmp/trace-req-nomlx.txt
uv pip install --python .venv/bin/python --no-deps "$(grep '^tabpfn==' requirements.txt | cut -d' ' -f1)"
uv pip install --python .venv/bin/python \
  safetensors einops huggingface-hub pydantic pydantic-settings filelock lightgbm

echo "==> editable install of the package"
uv pip install --python .venv/bin/python -e . --no-deps

# Local is pinned to CPU. TabPFN requires torch>=2.6 for MPS but the pin is 2.5.1
# (observed: ValueError 'TabPFN requires PyTorch >= 2.6 for MPS'). Local is for
# validation, not speed, and CPU keeps dtype at float64, matching the pod.
export TRACE_DEVICE=cpu

echo "==> smoke test"
.venv/bin/python scripts/verify_env.py

cat <<'MSG'

Next:
  source .venv/bin/activate
  export TRACE_DEVICE=cpu       # MPS requires upgrading to torch>=2.6
MSG
