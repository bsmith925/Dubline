#!/usr/bin/env bash
# Run the core suite under an environment override, restoring the server config afterwards.
#   eval/run_experiment.sh <suite.yaml> "<notes>" [KEY=VALUE ...]
# Each KEY=VALUE is applied to .env for the duration of the run (server restarted before and after).
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE="$1"; NOTES="$2"; shift 2
cp .env .env.experiment-backup
SEED_ARGS=()
for kv in "$@"; do
    key="${kv%%=*}"
    case "$key" in
        SEED_BUNDLE) SEED_ARGS+=(--seed-bundle "${kv#*=}"); continue;;
        SEED_TIER)   SEED_ARGS+=(--seed-tier "${kv#*=}"); continue;;
    esac
    if grep -q "^${key}=" .env; then sed -i "s|^${key}=.*|${kv}|" .env; else echo "$kv" >> .env; fi
done
systemctl --user restart dubline; sleep 5
cancel_jobs() {
    # Cancel queued/processing jobs so a killed run cannot leave orphans ahead of the next one.
    curl -s http://127.0.0.1:8000/api/jobs | python3 -c 'import json,sys
for j in json.load(sys.stdin):
    if j.get("status") in ("queued","processing"): print(j["id"])' | while read -r id; do
        curl -s -X POST "http://127.0.0.1:8000/api/jobs/$id/control/cancel" >/dev/null; done
}
trap 'cancel_jobs; cp .env.experiment-backup .env; systemctl --user restart dubline' EXIT
vendor/index-tts/.venv/bin/python -m eval.dubline_eval.cli run --suite "$SUITE" --notes "$NOTES" "${SEED_ARGS[@]}"
