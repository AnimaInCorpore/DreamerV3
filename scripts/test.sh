#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /c/msys64/opt/dreamerv3-python/python.exe -c 'import sys,unittest; sys.path.insert(0,"."); suite=unittest.defaultTestLoader.discover("tests"); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())'
