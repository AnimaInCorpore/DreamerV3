#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Override when another compatible Python (3.10-3.13) is preferred.
runtime_python="${DREAMERV3_PYTHON:-python3.11}"
if [[ ! -x .venv/bin/python ]]; then
  "$runtime_python" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
