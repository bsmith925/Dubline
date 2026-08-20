#!/usr/bin/env bash
# Start the Dubline server (Linux).  Binds to all interfaces by default so a
# remote client can submit videos; override with DUB_HOST / DUB_PORT.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
RUNTIME="$PROJECT_ROOT/vendor/index-tts/.venv/bin/python"
if [[ ! -x "$RUNTIME" ]]; then
    echo "The CUDA runtime is not installed. Run ./setup.sh once first." >&2
    exit 1
fi
export PYTHONUTF8=1
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$HOME/.local/bin:$PATH"
# Host/port come from the environment or .env via app.config (pydantic-settings).
read -r HOST PORT < <("$RUNTIME" -c "from app.config import settings; print(settings.dub_host, settings.dub_port)")
exec "$RUNTIME" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
