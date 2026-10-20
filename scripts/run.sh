#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=/c/msys64/opt/dreamerv3-python/python.exe
if [[ ! -x "$PYTHON" ]]; then
  echo 'Run powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 first.' >&2
  exit 1
fi
exec "$PYTHON" -c 'import runpy,sys; sys.path.insert(0,"."); runpy.run_module("dreamerv3.train",run_name="__main__")' "$@"
