# actions.py — custom action(s) for guardrails/rails/output.co.
#
# NeMo Guardrails auto-discovers an actions.py in the config directory.
# Pure Colang 1.0 expressions cannot do regex substitution, so sensitive-info
# *redaction* (as opposed to a simple block) needs a small registered Python
# action instead of being expressible in the .co files alone.
#
# NOTE: the exact key used to read the current bot response out of the
# action context ("bot_message" below) should be verified against the
# pinned nemoguardrails==0.14.1 action-context API during the deployment
# verification steps in README — this was written to the documented pattern
# at authoring time but action-context field names have shifted across
# NeMo Guardrails releases in the past.

import re

from nemoguardrails.actions import action

_PATTERNS = [
    # Generic API-key-like bearer tokens (sk-..., etc.)
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    # AWS access key IDs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # PEM private key blocks
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    # Email addresses
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Credit-card-like digit runs (13-19 digits, optionally grouped)
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
]


@action(name="filter_sensitive_info")
async def filter_sensitive_info(context: dict = None) -> dict:
    context = context or {}
    text = context.get("bot_message") or ""

    redacted = text
    found = False
    for pattern in _PATTERNS:
        if pattern.search(redacted):
            found = True
            redacted = pattern.sub("[REDACTED]", redacted)

    return {"redacted": redacted, "found": found}
