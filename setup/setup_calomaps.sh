#!/bin/bash
# CALOMAPS environment setup. Source this in a JupyterLab terminal:
#     source ~/setup_calomaps.sh
#
# What it does:
#   1. Loads Key4hep 2026-02-01 from CVMFS (Geant4, DD4hep, uproot, ROOT, PyTorch CPU)
#   2. Ensures the OpenGL workaround library (~/lib_hack) exists, then injects it
#   3. Makes sim/*.sh executable (the +x bit is lost on a fresh SSHFS clone)
#   4. Exports CALOMAPS_HOME and CALOMAPS_DATA_BASE, creating the data dir if needed
#   5. cd's into the simulation working directory
#
# Safe to source repeatedly. It only *warns* (never `exit`s) on a problem, so a
# missing piece won't kill your interactive shell.

# --- 1. Key4hep from CVMFS -------------------------------------------------
# Key4hep release pin. Don't bump casually: new releases occasionally change
# library ABI (e.g. torch major versions, ROOT minor versions) and can break
# saved-model loading or notebook output schemas. The current pin was tested
# end-to-end with the smoke sim and the dashboard regen on 2026-05-26.
# When you do bump (every ~6 months is a reasonable cadence): re-run
#   1. `ddsim ... -N 10 ...`     (smoke sim, see handbook.md §8)
#   2. notebooks/02_data_extraction.ipynb on a small subset
#   3. analysis/verify_ensembles.py end-to-end
# and confirm no regressions before merging the new pin.
KEY4HEP_RELEASE="2026-02-01"
if [ ! -d "/cvmfs/sw.hsf.org/key4hep" ]; then
  echo "WARNING: /cvmfs/sw.hsf.org/key4hep not found — is CVMFS mounted? (EAF images have it.)"
fi
echo "Loading Key4hep environment ($KEY4HEP_RELEASE)..."
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r "$KEY4HEP_RELEASE"

# --- 2. OpenGL workaround lib (~/lib_hack) ---------------------------------
# ddsim links libOpenGL.so.0; AlmaLinux 9 ships GL as libGL.so.1. Without this shim
# ddsim dies at startup with "libOpenGL.so.0: cannot open shared object file". We
# create the symlink automatically the first time, so it's one less manual step.
if [ ! -e "$HOME/lib_hack/libOpenGL.so.0" ]; then
  for _gl in /usr/lib64/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so.1; do
    if [ -e "$_gl" ]; then
      mkdir -p "$HOME/lib_hack"
      ln -sf "$_gl" "$HOME/lib_hack/libOpenGL.so.0"
      echo "Created OpenGL shim: ~/lib_hack/libOpenGL.so.0 -> $_gl"
      break
    fi
  done
  if [ ! -e "$HOME/lib_hack/libOpenGL.so.0" ]; then
    echo "WARNING: couldn't find libGL.so.1 to build ~/lib_hack/libOpenGL.so.0 — ddsim may fail to start."
  fi
fi
echo "Injecting OpenGL library hack..."
export LD_LIBRARY_PATH="$HOME/lib_hack:$LD_LIBRARY_PATH"

# --- 2b. CPU Jupyter kernel for notebooks 00/01/02 -------------------------
# A JupyterLab GUI kernel is launched by the notebook server and never sources
# this script, so on its own it has NO Key4hep stack (uproot/awkward/numpy) on
# PYTHONPATH. Register a kernel whose launcher *sources* Key4hep first — same
# pattern as setup_gpu_kernel.sh's GPU kernel. Rewritten on every source so it
# tracks the release pin above.
# Resolve the project root + data dir NOW (before the kernel wrapper is written) so the wrapper
# can bake CALOMAPS_HOME in: a GUI-launched Jupyter kernel does NOT inherit this terminal's
# environment, and without CALOMAPS_HOME the notebooks cannot put analysis/ on sys.path (the
# "No module named decal_cbfit" error). Override by exporting either before sourcing.
export CALOMAPS_HOME="${CALOMAPS_HOME:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
export CALOMAPS_DATA_BASE="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"

_CPU_KDIR="$HOME/.local/share/jupyter/kernels/calomaps_cpu"
_cpu_was_new=0; [ -f "$_CPU_KDIR/kernel.json" ] || _cpu_was_new=1
mkdir -p "$_CPU_KDIR"
cat > "$_CPU_KDIR/wrapper.sh" <<EOF
#!/bin/bash
# Source the Key4hep stack so uproot/awkward/numpy import in a GUI-launched kernel.
# unset the guard first: if this kernel is spawned FROM an already-sourced terminal
# (e.g. nbconvert), setup.sh would otherwise short-circuit and set no paths.
unset KEY4HEP_STACK
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r "$KEY4HEP_RELEASE" >/dev/null 2>&1
# a GUI kernel does not inherit the sourcing terminal's env, so bake the project paths in:
export CALOMAPS_HOME="$CALOMAPS_HOME"
export CALOMAPS_DATA_BASE="$CALOMAPS_DATA_BASE"
exec python -m ipykernel_launcher "\$@"
EOF
chmod +x "$_CPU_KDIR/wrapper.sh"
cat > "$_CPU_KDIR/kernel.json" <<EOF
{
 "argv": ["$_CPU_KDIR/wrapper.sh", "-f", "{connection_file}"],
 "display_name": "Key4hep (CPU)",
 "language": "python"
}
EOF
[ "$_cpu_was_new" = 1 ] && echo "Registered Jupyter kernel 'Key4hep (CPU)' for nb00/01/02 (reload JupyterLab to see it)."

# --- 3. Project root + executable sim scripts ------------------------------
# Locate the repo root from THIS script's own path (resolved through the
# ~/setup_calomaps.sh symlink), so it works wherever you cloned -- $HOME, /nashome,
# anywhere. Override by exporting CALOMAPS_HOME before sourcing.
export CALOMAPS_HOME="${CALOMAPS_HOME:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
# Make the sim driver scripts executable. git usually preserves the +x bit, but
# some filesystems / mirrored snapshots can drop it, so restore it as a safety net.
if [ -d "${CALOMAPS_HOME}/sim" ]; then
  chmod +x "${CALOMAPS_HOME}"/sim/*.sh 2>/dev/null || true
fi

# --- 4. Data directory -----------------------------------------------------
# Where the simulation output datasets live. Override by exporting before sourcing.
export CALOMAPS_DATA_BASE="${CALOMAPS_DATA_BASE:-$HOME/CALOMAPS-data}"
mkdir -p "$CALOMAPS_DATA_BASE" 2>/dev/null || true

# --- 5. Go to the sim working directory ------------------------------------
echo "Navigating to simulation working directory..."
cd "${CALOMAPS_HOME}/sim" 2>/dev/null \
  || echo "WARNING: ${CALOMAPS_HOME}/sim not found — is CALOMAPS_HOME set correctly?"

echo
echo "Environment ready. You can now run ddsim."
echo "  CALOMAPS_HOME      = $CALOMAPS_HOME"
echo "  CALOMAPS_DATA_BASE = $CALOMAPS_DATA_BASE"
echo "  Key4hep release    = $KEY4HEP_RELEASE"
echo "  cwd                = $(pwd)"
