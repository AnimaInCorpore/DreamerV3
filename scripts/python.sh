#!/usr/bin/env bash
# Shared interpreter selection; source after changing to the project root.
if [[ -n "${DREAMERV3_PYTHON:-}" ]]; then
  runtime_python="$DREAMERV3_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  runtime_python="$PWD/.venv/bin/python"
elif [[ -x /c/msys64/opt/dreamerv3-python/python.exe ]]; then
  runtime_python=/c/msys64/opt/dreamerv3-python/python.exe
else
  echo 'Run bash scripts/setup.sh (macOS/Linux) or scripts/setup.ps1 (Windows) first.' >&2
  exit 1
fi
