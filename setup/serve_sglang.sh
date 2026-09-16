#!/usr/bin/env bash
# Serve a model on this instance's GPU(s) with SGLang, OpenAI-compatible.
#
#   bash setup/serve_sglang.sh
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct bash setup/serve_sglang.sh
#   PORT=8001 API_KEY=secret bash setup/serve_sglang.sh
#
# vLLM or SGLang?
#   Both speak the same OpenAI API and either works here. SGLang tends to be
#   faster under many concurrent requests, which is the regime ShinkaEvolve
#   runs in when max_parallel_jobs is above 1. vLLM tends to be less fussy to
#   install. If you have no preference, start with vLLM; switch to SGLang if
#   proposal throughput is your bottleneck.
#
# Reasoning models: if the model emits chain-of-thought in a separate channel,
# pass --reasoning-parser (e.g. deepseek-r1, qwen3, glm45) so the content field
# holds the answer rather than the thinking. Shinka reads the content field.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-}"
API_KEY="${API_KEY:-}"
# SGLang captures CUDA graphs on startup too, and qBraid's GPU containers do not
# permit capture -- the same cudaErrorNotPermitted that kills vLLM. Disable it by
# default; set DISABLE_CUDA_GRAPH=0 on a host that does permit capture.
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: no nvidia-smi on PATH. This needs to run on a GPU instance." >&2
  echo "       Launch one from the On-Demand tab at https://account.qbraid.com/dashboard" >&2
  exit 1
fi

if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "error: sglang is not installed. Install it with:" >&2
  echo "         pip install 'sglang[all]'" >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
TP="${TP_SIZE:-1}"
if [ -z "${TP_SIZE:-}" ]; then
  TP=1
  while [ $((TP * 2)) -le "$GPU_COUNT" ]; do TP=$((TP * 2)); done
fi

echo "model              : $MODEL  (served as '$SERVED_NAME')"
echo "GPUs visible       : $GPU_COUNT (tp-size=$TP)"
echo "context length     : $CONTEXT_LENGTH"
echo "cuda graphs        : ${DISABLE_CUDA_GRAPH:+disabled}${DISABLE_CUDA_GRAPH:-enabled}"
echo "listening on       : http://${HOST}:${PORT}/v1"
echo "auth               : ${API_KEY:+Bearer <API_KEY>}${API_KEY:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/                     /'
echo

ARGS=(
  --model-path "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP"
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
)
[ "$DISABLE_CUDA_GRAPH" != "0" ] && ARGS+=(--disable-cuda-graph)
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")

exec python -m sglang.launch_server "${ARGS[@]}"
