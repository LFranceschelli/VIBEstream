#!/usr/bin/env bash
# VIBEstream installer for Linux:  bash install_linux.sh   [python executable]
# Creates .venv here and installs VIBEstream + its Python packages.
# The camera SDK (OpenEB / Metavision) is NOT installed here: see README.md.
# --system-site-packages keeps the Metavision Python modules importable.
set -euo pipefail
cd "$(dirname "$0")"
PY="${1:-python3}"
"$PY" -c "import sys; print('Using', sys.version.split()[0], sys.executable)"
"$PY" -c "import tkinter" 2>/dev/null || echo "WARNING: tkinter missing (the GUI needs it): e.g. sudo apt install python3-tk"
[ -x .venv/bin/python ] || "$PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[fast]"
.venv/bin/python -c "import numpy, scipy, cv2, h5py, vibe, vibe_hr; print('VIBEstream and core packages: OK')"
.venv/bin/python -c "import metavision_hal, metavision_core; print('Metavision: OK')" 2>/dev/null \
  || echo "Metavision: NOT importable - install OpenEB / Metavision SDK (camera and .raw files)."
echo "Done. Start the GUI with:  .venv/bin/vibestream    (or: .venv/bin/python EBIV_Main.py --gui)"
