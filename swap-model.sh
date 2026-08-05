#!/bin/bash
# Load a model on the llama-turboq router server (no restart needed).
#
# Usage:
#   swap-model.sh <model-name> [port]
#
# Model names are the sections in /home/jack/llm/llama-turboq/models.ini,
# e.g. Qwopus3.6-27B-v2-MTP, Qwen3.6-27B-MTP-pi-reasoning, Qwen3.6-27B,
# Qwen3.6-35B-A3B-UD, Ornith-1.0-9B, Gemma-4-12B.
#
# POST /models/load is synchronous: the router spawns/stops the child server
# and returns {"success": true} once the model is loaded (or an error). This
# script then polls GET /models until the requested model reports loaded.

set -euo pipefail

NAME="${1:?usage: swap-model.sh <model-name> [port]}"
PORT="${2:-8081}"
BASE="http://127.0.0.1:$PORT"

RESP=$(curl -s -m 900 -X POST "$BASE/models/load" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\"}")

STATUS=$(printf '%s' "$RESP" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("parse-error")
    sys.exit(0)
# router success payload: {"success": true}
if d.get("success"):
    print("loaded")
else:
    print("error")
')

if [ "$STATUS" = "error" ]; then
    echo "model swap FAILED:"
    printf '%s\n' "$RESP" | python3 -m json.tool
    exit 1
fi

# poll until the requested model reports loaded (the router load is
# synchronous, so this should be immediate; kept for robustness)
for _ in $(seq 1 60); do
    LOADED=$(curl -s -m 10 "$BASE/models" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('no')
    sys.exit(0)
for m in d.get('data', []):
    st = m.get('status')
    if isinstance(st, dict):
        st = st.get('value')
    if m.get('id') == '$NAME' and st == 'loaded':
        print('yes')
        sys.exit(0)
print('no')
")
    if [ "$LOADED" = "yes" ]; then
        echo "model '$NAME' is now loaded"
        exit 0
    fi
    sleep 2
done

echo "model '$NAME': loaded but status not confirmed via /models after 120s"
exit 1
