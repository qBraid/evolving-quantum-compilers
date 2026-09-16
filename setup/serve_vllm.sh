#!/usr/bin/env bash
# Serve a model on this instance's GPU(s) with vLLM, OpenAI-compatible.
#
#   bash setup/serve_vllm.sh                      # defaults below
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct bash setup/serve_vllm.sh
#   PORT=8001 API_KEY=secret bash setup/serve_vllm.sh
#
# Leave this running and drive it from a second terminal. The server prints
# "Application startup complete" when it is ready; first start also downloads
# the weights, which is usually the slow part.
#
# Model size vs GPU memory, roughly, for bf16 weights plus KV cache:
#
#     24 GB  (RTX 4090, L4)            7B comfortably, 14B quantised
#     48 GB  (L40S, RTX 6000 Ada)      14B comfortably, 32B quantised
#     80 GB  (A100 80GB, H100, H200)   32B comfortably
#
# Size up rather than down if you can. ShinkaEvolve asks the model to emit
# exact SEARCH/REPLACE diff blocks against the seed program, and small models
# are unreliable at that -- below about 14B a large fraction of proposals fail
# to parse and the run stalls without ever erroring out. See docs/CHOOSING_A_MODEL.md.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
API_KEY="${API_KEY:-}"
# CUDA graph capture is not permitted inside qBraid's GPU containers. vLLM dies
# during capture with
#     torch.AcceleratorError: CUDA error: operation not permitted
# from vllm/compilation/cuda_graph.py, and the server never comes up at all.
# --enforce-eager skips capture: it costs some decode throughput and is the
# difference between a server and no server. Set ENFORCE_EAGER=0 on a host that
# does permit capture.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: no nvidia-smi on PATH. This needs to run on a GPU instance." >&2
  echo "       Launch one from the On-Demand tab at https://account.qbraid.com/dashboard" >&2
  exit 1
fi

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "error: vllm is not installed. Install it with:" >&2
  echo "         pip install vllm" >&2
  exit 1
fi

# One vLLM worker per visible GPU. Tensor parallelism needs a divisor of the
# attention head count; powers of two are always safe, so round down to one.
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
TP="${TENSOR_PARALLEL_SIZE:-1}"
if [ -z "${TENSOR_PARALLEL_SIZE:-}" ]; then
  TP=1
  while [ $((TP * 2)) -le "$GPU_COUNT" ]; do TP=$((TP * 2)); done
fi

echo "model              : $MODEL"
echo "GPUs visible       : $GPU_COUNT (tensor-parallel-size=$TP)"
echo "context length     : $MAX_MODEL_LEN"
echo "cuda graphs        : ${ENFORCE_EAGER:+disabled (--enforce-eager)}${ENFORCE_EAGER:-enabled}"
echo "listening on       : http://${HOST}:${PORT}/v1"
echo "auth               : ${API_KEY:+Bearer <API_KEY>}${API_KEY:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/                     /'
echo

ARGS=(
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
[ "$ENFORCE_EAGER" != "0" ] && ARGS+=(--enforce-eager)
# Without a key the endpoint is open to anything that can reach the port. That
# is fine inside a single-user qBraid instance and not fine if you expose it.
[ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")

exec vllm serve "${ARGS[@]}"
