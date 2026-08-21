#!/usr/bin/env bash
# Run the core suite under an environment override, restoring the server config afterwards.
#   eval/run_experiment.sh <suite.yaml> "<notes>" [KEY=VALUE ...]
# Each KEY=VALUE is applied to .env for the duration of the run (server restarted before and after).
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE="$1"; NOTES="$2"; shift 2
cp .env .env.experiment-backup
for kv in "$@"; do
    key="${kv%%=*}"
    if grep -q "^${key}=" .env; then sed -i "s|^${key}=.*|${kv}|" .env; else echo "$kv" >> .env; fi
done
systemctl --user restart dubline; sleep 5
trap 'cp .env.experiment-backup .env; systemctl --user restart dubline' EXIT
vendor/index-tts/.venv/bin/python -m eval.dubline_eval.cli run --suite "$SUITE" --notes "$NOTES"
