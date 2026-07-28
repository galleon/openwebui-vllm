#!/usr/bin/env bash
# Full benchmark sweep — supports multiple hardware targets.
# Usage: bash run_sweep.sh [TARGET [PHASE]]
#   TARGET: dgx-spark-gb10 (default) | rtx-pro-6000 | l40s
#   PHASE:  all (default) | gemma4 | nemotron | qwen
#
# For each model phase, two sub-sweeps are run back-to-back:
#   1. Without guardrails  (Open WebUI → vLLM direct)
#   2. With guardrails     (Open WebUI → guardrails proxy → nemoguardrails → vLLM)
#
# Results land in results/<TARGET>/<model>/   and   results/<TARGET>/<model>-rails/
#
# Prerequisites:
#   - .env already configured for the target hardware (cp .env.<TARGET>.* .env)
#   - OPENWEBUI_API_KEY and OPENWEBUI_KB_ID set in .env
#   - vLLM container healthy before running
#   - uv installed
#   - For guardrails sub-sweeps: guardrails image already built
#       docker compose build guardrails

set -euo pipefail
cd "$(dirname "$0")"

TARGET=${1:-dgx-spark-gb10}
PHASE=${2:-all}

case "$TARGET" in
  dgx-spark-gb10)
    GPU_MEM=0.70
    GEMMA4_MAX_LEN=8192
    NEMOTRON_MAX_LEN=8192
    QWEN_MAX_LEN=32768   # Qwen3.5 think chains overflow at 8192
    USER_LADDER=(10 20 30 50 100)
    RAMP_LADDER=(2  2  5  5  10)
    ;;
  rtx-pro-6000)
    GPU_MEM=0.55
    GEMMA4_MAX_LEN=16384
    NEMOTRON_MAX_LEN=16384
    QWEN_MAX_LEN=16384
    USER_LADDER=(10 20 30 50 100)
    RAMP_LADDER=(2  2  5  5  10)
    ;;
  l40s)
    GPU_MEM=0.90
    GEMMA4_MAX_LEN=8192
    NEMOTRON_MAX_LEN=8192
    QWEN_MAX_LEN=8192    # KV budget ~8 GB on L40S — keep context short
    USER_LADDER=(5 10 15 20 30)
    RAMP_LADDER=(1  2  2  2   5)
    ;;
  *)
    echo "Unknown target: $TARGET. Valid values: dgx-spark-gb10, rtx-pro-6000, l40s" >&2
    exit 1
    ;;
esac

WEBUI_PORT=$(grep '^WEBUI_PORT=' .env | cut -d= -f2)
HOST="http://localhost:${WEBUI_PORT:-3000}"
LOCUST="./locustfile.py"

# ── helpers ──────────────────────────────────────────────────────────────────

# Safe upsert: update key if present, append if missing.
upsert_env() {
    local key=$1 val=$2
    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${val}|" .env
    else
        echo "${key}=${val}" >> .env
    fi
}

wait_healthy() {
    local svc=${1:-vllm}
    echo "[sweep] waiting for $svc to be healthy..."
    until [ "$(docker inspect --format='{{.State.Health.Status}}' "$svc" 2>/dev/null)" = "healthy" ]; do
        sleep 15
    done
    echo "[sweep] $svc is healthy"
}

smoke_test() {
    local model api_key http_code
    model=$(grep '^OPENWEBUI_MODEL=' .env | cut -d= -f2)
    api_key=$(grep '^OPENWEBUI_API_KEY=' .env | cut -d= -f2)
    echo "[sweep] smoke-testing $HOST with model $model ..."
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$HOST/api/chat/completions" \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"stream\":false}" \
        --max-time 60 2>/dev/null || echo "000")
    if [[ "$http_code" != "200" ]]; then
        echo "[sweep] ERROR: smoke test got HTTP $http_code (model=$model) — aborting sweep" >&2
        exit 1
    fi
    echo "[sweep] smoke test passed (HTTP 200)"
}

warmup() {
    # Two full-length requests trigger any lazy kernel compilation (critical for
    # FP4 models: FlashInfer FP4 MoE profiling on the first real batch can block
    # generation for several minutes without this).
    local model api_key http_code i
    model=$(grep '^OPENWEBUI_MODEL=' .env | cut -d= -f2)
    api_key=$(grep '^OPENWEBUI_API_KEY=' .env | cut -d= -f2)
    echo "[sweep] warming up $model (2 requests, up to 10 min each) ..."
    for i in 1 2; do
        echo "[sweep] warmup $i/2 ..."
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST "$HOST/api/chat/completions" \
            -H "Authorization: Bearer $api_key" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What spectator areas are available at the Nürburgring 24h race?\"}],\"stream\":false}" \
            --max-time 600 2>/dev/null || echo "000")
        echo "[sweep] warmup $i/2: HTTP $http_code"
    done
    echo "[sweep] warmup done"
}

run_locust() {
    local tag=$1 users=$2 ramp=$3 outdir=$4
    local prefix="${tag}_u${users}"
    local tag_args=(); [[ "$tag" != "mixed" ]] && tag_args=(-T "$tag")
    echo "[sweep] $(date '+%H:%M:%S') ▶ $prefix (tag=$tag, users=$users)"
    uv run "$LOCUST" --host "$HOST" \
        --headless -u "$users" -r "$ramp" --run-time 5m --reset-stats \
        --csv="$outdir/$prefix" "${tag_args[@]}" \
        >"$outdir/${prefix}.stdout.log" 2>>"$outdir/${prefix}.stderr.log"
    echo "[sweep] $(date '+%H:%M:%S') ✓ $prefix done"
}

sweep() {
    local outdir=$1
    mkdir -p "$outdir"
    for tag in nothink think mixed; do
        for i in "${!USER_LADDER[@]}"; do
            run_locust "$tag" "${USER_LADDER[$i]}" "${RAMP_LADDER[$i]}" "$outdir"
        done
    done
}

# switch_model: update .env, restart vLLM + Open WebUI, smoke-test, warmup.
# Args: model tool_call_parser parser_plugin parser fp4 fp4_backend max_len
switch_model() {
    local model=$1 tool_call_parser=$2 parser_plugin=$3 parser=$4 \
          fp4=$5 fp4_backend=$6 max_len=$7
    echo "[sweep] switching vLLM to $model (MAX_MODEL_LEN=$max_len)"
    upsert_env VLLM_MODEL                  "$model"
    upsert_env OPENWEBUI_MODEL             "$model"
    upsert_env VLLM_TOOL_CALL_PARSER       "$tool_call_parser"
    upsert_env VLLM_REASONING_PARSER_PLUGIN "$parser_plugin"
    upsert_env VLLM_REASONING_PARSER       "$parser"
    upsert_env VLLM_USE_FLASHINFER_MOE_FP4 "$fp4"
    upsert_env VLLM_FLASHINFER_MOE_BACKEND "$fp4_backend"
    upsert_env VLLM_GPU_MEMORY_UTILIZATION "$GPU_MEM"
    upsert_env VLLM_MAX_MODEL_LEN          "$max_len"
    upsert_env VLLM_KV_CACHE_DTYPE         fp8
    docker compose up -d vllm
    wait_healthy vllm
    docker compose up -d open-webui
    wait_healthy open-webui
    smoke_test
    warmup
}

ensure_qdrant() {
    if ! docker inspect qdrant --format='{{.State.Status}}' 2>/dev/null | grep -q running; then
        echo "[sweep] starting Qdrant ..."
        docker compose --profile qdrant up -d qdrant
        until [ "$(docker inspect --format='{{.State.Health.Status}}' qdrant 2>/dev/null)" = "healthy" ]; do
            sleep 5
        done
        echo "[sweep] Qdrant healthy"
    fi
}

enable_guardrails() {
    echo "[sweep] enabling guardrails (Open WebUI → guardrails:8001 → vLLM)"
    upsert_env OPENAI_API_BASE_URL "http://guardrails:8001/v1"
    # Recreate guardrails so it picks up the current VLLM_MODEL from .env
    docker compose --profile guardrails up -d guardrails
    wait_healthy guardrails
    docker compose up -d open-webui
    wait_healthy open-webui
    smoke_test
}

disable_guardrails() {
    echo "[sweep] disabling guardrails (Open WebUI → vLLM direct)"
    upsert_env OPENAI_API_BASE_URL "http://vllm:8000/v1"
    docker compose --profile guardrails stop guardrails 2>/dev/null || true
    docker compose up -d open-webui
    wait_healthy open-webui
}

# run_model_phase: no-rails sweep then rails sweep for the current model.
# Args: outdir_base (e.g. results/dgx-spark-gb10/gemma4-nvfp4)
run_model_phase() {
    local base=$1

    echo "[sweep] ── no-guardrails sweep → $base ──"
    disable_guardrails
    warmup
    sweep "$base"

    echo "[sweep] ── guardrails sweep → ${base}-rails ──"
    enable_guardrails
    warmup
    sweep "${base}-rails"
    disable_guardrails   # restore clean state for next model switch
}

# ── Entry point ───────────────────────────────────────────────────────────────

ensure_qdrant

# ── Phase 0: Gemma-4-26B-A4B-NVFP4 ──────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "gemma4" ]]; then
    echo "=== [$TARGET] Phase 0: Gemma-4-26B-A4B-NVFP4 ==="
    switch_model \
        "nvidia/Gemma-4-26B-A4B-NVFP4" \
        "gemma4" \
        "/vllm_plugins/noop.py" \
        "gemma4" \
        "0" \
        "throughput" \
        "$GEMMA4_MAX_LEN"
    run_model_phase "results/$TARGET/gemma4-nvfp4"
fi

# ── Phase 1: Nemotron-3-Nano-30B-A3B-NVFP4 ───────────────────────────────────

if [[ "$TARGET" != "l40s" ]] && [[ "$PHASE" == "all" || "$PHASE" == "nemotron" ]]; then
    echo "=== [$TARGET] Phase 1: Nemotron-3-Nano-30B-A3B-NVFP4 ==="
    switch_model \
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
        "qwen3_coder" \
        "/vllm_plugins/nano_v3_reasoning_parser.py" \
        "nano_v3" \
        "1" \
        "throughput" \
        "$NEMOTRON_MAX_LEN"
    run_model_phase "results/$TARGET/nemotron-nano"
fi

# ── Phase 2: Qwen3.5-35B-A3B ─────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "qwen" ]]; then
    if [[ "$TARGET" == "l40s" ]]; then
        QWEN_MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
        QWEN_TCP="qwen3_coder"
        QWEN_FP4="0"
        QWEN_FP4_BACKEND="throughput"
        QWEN_OUTDIR="results/$TARGET/qwen3.5-35b-fp8"
    else
        QWEN_MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
        QWEN_TCP="qwen3_coder"
        QWEN_FP4="1"
        QWEN_FP4_BACKEND="throughput"
        QWEN_OUTDIR="results/$TARGET/qwen3.6-35b-nvfp4"
    fi

    echo "=== [$TARGET] Phase 2: $QWEN_MODEL ==="
    switch_model \
        "$QWEN_MODEL" \
        "$QWEN_TCP" \
        "/vllm_plugins/noop.py" \
        "qwen3" \
        "$QWEN_FP4" \
        "$QWEN_FP4_BACKEND" \
        "$QWEN_MAX_LEN"
    run_model_phase "$QWEN_OUTDIR"
fi

# ── Phase 3: restore Gemma-4 (default production model) ──────────────────────

if [[ "$PHASE" == "all" ]] && [[ "$TARGET" != "l40s" ]]; then
    echo "=== [$TARGET] Phase 3: restoring Gemma-4 ==="
    switch_model \
        "nvidia/Gemma-4-26B-A4B-NVFP4" \
        "gemma4" \
        "/vllm_plugins/noop.py" \
        "gemma4" \
        "0" \
        "throughput" \
        "$GEMMA4_MAX_LEN"
fi

echo "=== sweep complete ==="
