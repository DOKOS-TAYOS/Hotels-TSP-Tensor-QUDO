#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[setup.sh] $1"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXTRAS="${1:-dev,ui,cudaq}"

log "Project root resolved to: ${PROJECT_ROOT}"
log "Requested extras: ${EXTRAS}"

if ! command -v python3 >/dev/null 2>&1; then
  log "ERROR: python3 was not found in PATH."
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  log "ERROR: Python 3.12+ is required."
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ -d ".venv" ]]; then
  log "Step 1/4: Reusing existing .venv"
else
  log "Step 1/4: Creating virtual environment in .venv"
  python3 -m venv .venv
fi

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
log "Step 2/4: Upgrading pip"
"${VENV_PYTHON}" -m pip install --upgrade pip

log "Step 3/4: Installing project dependencies"
if [[ -n "${EXTRAS}" ]]; then
  "${VENV_PYTHON}" -m pip install -e ".[${EXTRAS}]"
else
  "${VENV_PYTHON}" -m pip install -e .
fi

log "Step 4/4: Preparing environment file"
if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp ".env.example" ".env"
  log "Created .env from .env.example"
fi

log "Setup completed. Activate with: source .venv/bin/activate"
