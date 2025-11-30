#!/usr/bin/env bash
# run_all.sh
# Usage: ./run_all.sh
# This script will try to activate a virtualenv (looks for .venv, venv, env)
# then run the three python scripts in sequence. It exits on any error.

set -euo pipefail

echo "=> Searching for virtual environment to activate..."
VENV_CANDIDATES=(.venv venv env)
VENV_ACTIVATED=""

for d in "${VENV_CANDIDATES[@]}"; do
  if [ -d "$d" ] && [ -f "$d/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$d/bin/activate"
    VENV_ACTIVATED="$d"
    echo "=> Activated virtual environment: $d"
    break
  fi
done

if [ -z "$VENV_ACTIVATED" ]; then
  echo "=> No virtual environment found (looked for: ${VENV_CANDIDATES[*]})."
  echo "=> Continuing without activating a virtualenv. If you want activation,"
  echo "   create one with: python3 -m venv .venv"
fi

# Ensure python3 is available
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 2
fi

echo "=> Running: python3 src/fx_flow_model.py"
python3 src/fx_flow_model.py

echo "=> Running: python3 src/fz_flow_animation.py"
python3 src/fx_flow_animation.py

echo "=> Running: python3 src/backtest_fxmodel.py"
python3 src/backtest_fxmodel.py

echo "=> All done."
