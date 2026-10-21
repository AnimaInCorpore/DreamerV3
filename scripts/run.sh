#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/python.sh
exec "$runtime_python" -c 'import runpy,sys; sys.path.insert(0,"."); runpy.run_module("dreamerv3.train",run_name="__main__")' "$@"
