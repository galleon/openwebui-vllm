#!/bin/sh
# Renders the read-only mounted guardrails/ config (bind-mounted at /config)
# into a writable runtime directory, substituting the env vars NeMo
# Guardrails' config.yml does not expand natively, then execs the server.
set -eu

SRC_DIR=/config
RUNTIME_DIR=/tmp/guardrails-config

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

exec nemoguardrails server \
    --config="$RUNTIME_DIR" \
    --port 8001 \
    --default-config-id default \
    $VERBOSE_FLAG
