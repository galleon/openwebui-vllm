#!/bin/bash
# Renders the read-only mounted guardrails/ config (bind-mounted at /config)
# into a writable runtime directory, substituting the env vars NeMo
# Guardrails' config.yml does not expand natively, then execs the server.
set -eu

SRC_DIR=/config
CONFIGS_DIR=/tmp/guardrails-config
RUNTIME_DIR="$CONFIGS_DIR/default"

mkdir -p "$RUNTIME_DIR/rails"

# IMPORTANT: the explicit variable list below is deliberate, not cosmetic.
# .co (Colang) files use their own "$variable" syntax extensively (e.g.
# $user_message, $allowed). envsubst run WITHOUT an explicit list replaces
# every $VAR-shaped token it finds with the matching env var (or blanks it
# if unset) — which would silently corrupt every Colang flow. Passing an
# explicit list restricts substitution to only the names below.
SUBST_VARS='$VLLM_BASE_URL $VLLM_MODEL $GUARDRAILS_ALLOWED_TOPICS $GUARDRAILS_STREAMING_ENABLED'

envsubst "$SUBST_VARS" < "$SRC_DIR/config.yml" > "$RUNTIME_DIR/config.yml"

for f in "$SRC_DIR"/rails/*.co; do
    envsubst "$SUBST_VARS" < "$f" > "$RUNTIME_DIR/rails/$(basename "$f")"
done

# actions.py is plain Python — copy as-is, no substitution.
cp "$SRC_DIR/actions.py" "$RUNTIME_DIR/actions.py"

VERBOSE_FLAG=""
if [ "${GUARDRAILS_LOG_LEVEL:-info}" = "debug" ]; then
    VERBOSE_FLAG="--verbose"
fi

# Start nemoguardrails on the internal port (8002).
# The proxy (port 8001, below) routes /v1/models to vLLM and everything else
# here — Open WebUI only sees the proxy, which makes models discoverable even
# though nemoguardrails has no /v1/models endpoint of its own.
nemoguardrails server \
    --config="$CONFIGS_DIR" \
    --port 8002 \
    --default-config-id default \
    $VERBOSE_FLAG &
RAILS_PID=$!

# Wait for nemoguardrails to open its port before accepting proxy traffic.
echo "[entrypoint] waiting for nemoguardrails on port 8002 ..."
until bash -c '>/dev/tcp/localhost/8002' 2>/dev/null; do sleep 1; done
echo "[entrypoint] nemoguardrails ready, starting proxy on port 8001"

# Run the proxy in the foreground as the container's main process.
# If either process dies, the shell (PID 1) catches the exit via wait -n.
python3 /config/proxy.py &
PROXY_PID=$!

wait -n $RAILS_PID $PROXY_PID
echo "[entrypoint] a child process exited — shutting down"
kill $RAILS_PID $PROXY_PID 2>/dev/null
wait
