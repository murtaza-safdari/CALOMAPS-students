#!/bin/bash
# One-shot GPU kernel setup for notebook 03 (the DECAL ML / dashboard notebook).
#
# The CVMFS Key4hep stack ships a CPU-only PyTorch, so notebook 03 needs a venv
# with a CUDA build. This script builds a *self-contained* venv (CUDA torch +
# the few packages nb03 imports) and registers a "Key4hep + GPU" Jupyter kernel
# whose launcher uses a clean PYTHONPATH, so the venv's cu121 torch always wins
# (no CVMFS shadowing). nb00 / nb01 / nb02 don't need this — they use "Key4hep (CPU)".
#
# Usage (from a JupyterLab terminal, after `source ~/setup_calomaps.sh`):
#     bash setup/setup_gpu_kernel.sh
#
# By default the venv goes in /tmp (large, fast, but WIPED when the EAF container
# restarts — just re-run this script then). For a persistent install set
# CALOMAPS_GPU_ENV to a home path with ~5 GB free, e.g.:
#     CALOMAPS_GPU_ENV=$HOME/calomaps_gpu_env bash setup/setup_gpu_kernel.sh
set -e

VENV="${CALOMAPS_GPU_ENV:-/tmp/calomaps_gpu_env}"
# Prefer the Key4hep python on PATH (robust to release-pin bumps); fall back to the
# pinned 2026-02-01 interpreter if PATH isn't set up (e.g. setup_calomaps.sh not sourced).
if [ -n "${CALOMAPS_PYBIN:-}" ]; then
  PYBIN="$CALOMAPS_PYBIN"                          # explicit override: trust it as-is
else
  PYBIN="$(command -v python3.13 || true)"         # auto-detect the Key4hep python on PATH
  if [ -z "$PYBIN" ] || ! echo "$PYBIN" | grep -q cvmfs; then
    PYBIN=/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/python/3.13.8-z2dydk/bin/python3.13
  fi
fi
KDIR="$HOME/.local/share/jupyter/kernels/calomaps_gpu"

if [ ! -x "$PYBIN" ]; then
  echo "ERROR: Key4hep python not found at $PYBIN."
  echo "       Source the environment first:  source ~/setup_calomaps.sh"
  exit 1
fi

case "$VENV" in
  /tmp/*) echo "NOTE: venv is under /tmp ($VENV) — this is wiped on container restart."
          echo "      For a persistent kernel: CALOMAPS_GPU_ENV=\$HOME/calomaps_gpu_env bash setup/setup_gpu_kernel.sh" ;;
esac

echo "[1/4] Creating venv at $VENV (python: $PYBIN)"
"$PYBIN" -m venv --system-site-packages "$VENV"

# Install with a CLEAN PYTHONPATH so packages land in the venv (not shadowed by
# CVMFS) and torch's deps resolve to the venv copies.
echo "[2/4] Installing numpy / scipy / matplotlib / ipykernel / uproot / awkward"
TMPDIR=/tmp env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir --ignore-installed \
    numpy scipy matplotlib ipykernel uproot awkward
echo "[3/4] Installing CUDA (cu121) PyTorch (~4.4 GB; can take several minutes on a busy EAF)"
TMPDIR=/tmp env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

echo "[4/4] Registering the 'Key4hep + GPU' kernel"
# setup_calomaps.sh normally already exported these; resolve as a fallback so the wrapper below
# can bake them in (a GUI kernel does not inherit this terminal's environment).
CALOMAPS_HOME="${CALOMAPS_HOME:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
CALOMAPS_DATA_BASE="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"
mkdir -p "$KDIR"
cat > "$KDIR/wrapper.sh" <<EOF
#!/bin/bash
# clean PYTHONPATH -> the venv's cu121 torch wins over CVMFS's CPU torch
unset PYTHONPATH PYTHONHOME
# a GUI kernel does not inherit the sourcing terminal's env, so bake the project paths in:
export CALOMAPS_HOME="$CALOMAPS_HOME"
export CALOMAPS_DATA_BASE="$CALOMAPS_DATA_BASE"
exec "$VENV/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$KDIR/wrapper.sh"
cat > "$KDIR/kernel.json" <<EOF
{
 "argv": ["$KDIR/wrapper.sh", "-f", "{connection_file}"],
 "display_name": "Key4hep + GPU",
 "language": "python"
}
EOF

echo
echo "Verifying (this MUST report cuda True):"
env -u PYTHONPATH "$VENV/bin/python" - <<'PYEOF'
import sys
import torch
ok = torch.cuda.is_available()
print("  torch", torch.__version__, "| cuda", ok)
if not ok:
    print("  ERROR: CUDA not available after install. Are you on a GPU profile?")
    print("         Re-spawn an EAF GPU server, then re-run this script.")
    sys.exit(1)
PYEOF

echo
echo "Done. In JupyterLab, open notebooks/03_ml_training_and_eval.ipynb and pick"
echo "the 'Key4hep + GPU' kernel. (nb00 / nb01 / nb02 use 'Key4hep (CPU)'.)"
