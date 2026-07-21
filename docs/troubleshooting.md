# Troubleshooting

Infrastructure-level quirks that bite when running CALOMAPS on the Fermilab Elastic Analysis Facility (EAF). Each entry is a known pattern with a workaround.

For **project-level errors** (your `ddsim` exited non-zero, a notebook cell threw a NameError, etc.), see [handbook.md §14](handbook.md#14-common-gotchas-code-level) instead.

---

## SSHFS occasionally writes zero-byte files

*Only applies if you use the optional laptop SSHFS mount (handbook §16.1). On EAF itself, `/nashome` is native NFS and unaffected — skip this entry.*

**Symptom.** A `cp` (or git internal write, or `python -m venv`, or any small-file write) onto the `/nashome`-via-SSHFS mount succeeds, but the resulting file is **zero bytes** even though `ls` shows the correct size. Reading the file later reveals it's all NULs.

**Workaround.**

- For copying files: prefer `rsync -a` over `cp -r`. Rsync writes atomically (write-to-tempfile, rename) which dodges the issue on most attempts.
- For git operations: don't do `git init` / `git add` / `git commit` on the `/nashome`-mounted tree. Instead, rsync the working tree to a local-FS path (`/tmp/CALOMAPS-push/`), run git there, push from there, then `git clone` back from github to `/nashome` if you want a local checkout.
- After git operations on SSHFS, `git fsck --full` is cheap insurance against zero-byte objects.

**Diagnose.** `file path/to/suspect.xml` reports `data` (binary) instead of `ASCII text`; `xxd <file> | head` shows `00 00 00 …`.

---

## CVMFS Key4hep ships a CPU-only PyTorch

**Symptom.** In any environment that has the CVMFS Key4hep 2026-02-01 stack loaded, `torch.cuda.is_available()` returns `False` even on a GPU node. Worse, `torch.cuda.get_device_name(0)` raises `AssertionError: Torch not compiled with CUDA enabled` — the CVMFS torch was built without CUDA support at all.

**Workaround.** Install a CUDA-enabled torch into a venv that's *first* in your `sys.path`. See [handbook.md §11.2](handbook.md#112-training-new-models-on-the-gpu) for the full recipe. The short version:

```bash
/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/python/3.13.8-z2dydk/bin/python3.13 \
    -m venv --system-site-packages /tmp/cu_torch_env
/tmp/cu_torch_env/bin/pip install --force-reinstall torch \
    --index-url https://download.pytorch.org/whl/cu121
```

Then to actually use the cu121 torch (see next entry).

---

## CVMFS PYTHONPATH shadows your venv's torch

**Symptom.** After `pip install --force-reinstall torch+cu121` into a venv with `--system-site-packages = true`, `import torch` *still* loads the CVMFS CPU-only build. Even from the venv's own Python interpreter.

**Root cause.** When `source ~/setup_calomaps.sh` (or directly `source /cvmfs/.../setup.sh`) runs, it pre-pends ~30 paths to `PYTHONPATH`, including `/cvmfs/.../py-torch/.../site-packages`. That comes **earlier** in `sys.path` than the venv's own site-packages, so `import torch` finds CVMFS first.

**Workaround.** Drop CVMFS torch paths from `sys.path` and prepend the venv's site-packages before `import torch`:

```python
import sys
sys.path = [p for p in sys.path if "py-torch" not in p]
sys.path.insert(0, "/tmp/cu_torch_env/lib/python3.13/site-packages")
import torch    # now resolves to your venv-installed cu121 build
```

**Exception.** Inside a JupyterLab notebook using the `Key4hep + GPU` kernel, the kernel spawns without `setup_calomaps.sh` having been sourced — so the venv's torch wins naturally. No shim required in that path.

---

## JupyterHub `/api/kernels` POST returns 500 mid-session

**Symptom.** Spawning a Jupyter kernel via the JupyterHub REST API (`POST /user/<u>/api/kernels`) returns `HTTP 500 Unhandled error` with an empty traceback. The user-server itself is healthy: GET on `/api/contents` works, terminals work, files are accessible — only kernel-spawn fails.

**Workaround.**

- If you're driving things programmatically and need code execution: open a **terminal** via `POST /user/<u>/api/terminals` (which stays healthy), then connect via WebSocket and run shell commands. A Python invocation as a subprocess of the terminal shell works for code execution without a Jupyter kernel.
- If you can open JupyterLab in a browser: just open a notebook normally — the kernel-spawn path through the UI is different and usually unaffected.
- Last resort: restart your user-server from the JupyterHub control panel ("Stop My Server" → "Start My Server"). This wipes container-local paths like `/tmp/cu_torch_env`, so re-do any installs.

---

## JupyterHub `/api/contents` PUT returns 500

**Symptom.** `PUT /user/<u>/api/contents/<path>` (used to programmatically write files) returns `HTTP 500` while `GET` on the same endpoint works. The home directory may be quota-pressured (see next entry).

**Workaround.** Write the file via a terminal session instead: `POST /user/<u>/api/terminals` to create a terminal, connect over WebSocket, send a command like `base64 -d <<< '<base64-of-content>' > /path/to/file`.

---

## `/home/<username>` on EAF fills up

**Symptom.** `df -h /home/$USER` shows 100% used, 0 bytes free. Subsequent writes anywhere under `/home` fail with `disk quota exceeded`, including small ones (a 1 KB symlink update).

**Diagnose.** `du -sh ~/.cache/* ~/.conda/* /home/$USER/* 2>/dev/null | sort -hr | head -10`

**Workaround.** The usual large-and-disposable culprits:
- `~/.cache/pip` — pip's wheel cache. Routinely 2-3 GB. Safe to delete: `rm -rf ~/.cache/pip`.
- `~/.conda/pkgs` — conda's package cache. Hundreds of MB. Safe to delete (`conda clean --packages` or `rm -rf ~/.conda/pkgs`).
- Old project data — large `.root` / `.npz` files left behind by previous runs. Delete them to free space — `/home` is your only quota for now.

If you want a persistent-but-isolated install of something big (e.g., a CUDA torch venv) and `/home` is tight, install to `/tmp/<name>` instead — EAF's `/tmp` is an overlay filesystem with several TB free. The cost: `/tmp` is wiped on container restart.

---

## macOS resource-fork files (`._*`) leak into `.git/`

**Symptom.** After `git clone` of a CALOMAPS repo onto a `/nashome` SSHFS mount, `git fsck --full` complains:

```
error: refs/heads/._main: badRefName: invalid refname format
error: refs/heads/._main: badRefContent:
error: refs/._heads: badRefName: invalid refname format
... etc
```

**Root cause.** macOS creates `._<name>` shadow files to store extended attributes on filesystems that don't support them natively. SSHFS surfaces these through the mount, and git's ref scanner picks them up as malformed ref names.

**Workaround.** They're harmless noise. Either ignore them, or sweep them out periodically:

```bash
find .git -name '._*' -type f -delete
```

(Same applies to any `._*` you find scattered through the work tree. The `.gitignore` already excludes them from being tracked.)

---

## `Key4hep + GPU` Jupyter kernel won't spawn via REST API

**Symptom.** `POST /user/<u>/api/kernels` with `{"name": "my_gpu_env"}` either returns 500 or returns 200 but the WebSocket connection drops immediately when you try to execute code.

**Root cause.** The `my_gpu_env` kernel.json points at `/home/<user>/my_gpu_env/bin/python` directly. That python expects CVMFS to be visible (because `my_gpu_env` inherits `--system-site-packages` from a CVMFS Python). When JupyterLab UI spawns the kernel, CVMFS is pre-set in the spawn environment. When the bare REST API spawns it, CVMFS is **not** pre-set, and the kernel fails on `import` of CVMFS-provided modules.

**Workaround.**

- For interactive notebook work: open the notebook in **JupyterLab UI**, pick the `Key4hep + GPU` kernel from the kernel selector. This path works.
- For programmatic code execution: use the **`Key4hep (CPU)`** kernel (different kernelspec, sources CVMFS in its own launcher) — it spawns cleanly from the REST API. Or use a terminal + subprocess, as in the API-500 workaround above.

---

## File mode bits (the `+x` executable flag) get lost via SSHFS on clone

**Symptom.** Immediately after `git clone` into `/nashome` you have an unstaged "modified" set:

```
modified:   sim/generate_batched.sh
modified:   sim/generate_dataset.sh
```

`git diff` shows no content change, just `old mode 100755 → new mode 100644`. The clone failed to preserve the executable bit.

**Workaround.** Re-add it manually after clone:

```bash
chmod +x sim/*.sh
```

Then `git status` is clean again. Permanent fix: `git config core.fileMode false` if you don't want git tracking file modes on this checkout at all — but that hides the issue rather than fixing it.

---

## "Key4hep + GPU" kernel dies on launch from nbconvert / freshly-spawned shells

**Symptom.** A GPU kernel whose `kernel.json` runs the venv python directly
(`~/my_gpu_env/bin/python -m ipykernel_launcher`) starts fine from the JupyterLab
UI but **dies immediately** when launched by `nbconvert` (or any non-interactive
spawn), with `No module named ipykernel_launcher` /
`Kernel died before replying to kernel_info`.

**Root cause.** `ipykernel` (and numpy/scipy/...) live in CVMFS prefixes that are
only added to `PYTHONPATH` by `source setup_calomaps.sh`. An interactive terminal
that sourced CVMFS earlier *looks* fine, but a kernel spawned with a clean
environment never gets those paths — and the venv (created `--system-site-packages`
from the CVMFS python, which does **not** carry ipykernel in its base
site-packages) can't find `ipykernel_launcher`.

**Workaround.** Make the GPU kernel **self-contained** and CVMFS-independent:
install everything the notebook needs *into* the venv with a clean `PYTHONPATH`,
and launch via a wrapper that clears `PYTHONPATH` so the venv's cu121 torch wins:

```bash
env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir --ignore-installed \
    numpy scipy matplotlib ipykernel
env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu121
# wrapper.sh:   unset PYTHONPATH PYTHONHOME; exec "$VENV/bin/python" -m ipykernel_launcher "$@"
```

`setup/setup_gpu_kernel.sh` does this, and additionally installs `uproot`/`awkward`
(the nb03 §10.2 capstone re-reads the raw `.root` files). The core training path
needs only torch/numpy/scipy/matplotlib, so a self-contained venv covers everything
without CVMFS at all. This supersedes the older `~/my_gpu_env` + `sys.path`-shim
recipe for the *notebook* path (the shim is still fine for headless scripts).

---

## Notebook can't import `uproot` (or find `CALOMAPS_HOME`) — you're on the wrong kernel

**Symptom.** A notebook cell fails with `ModuleNotFoundError: No module named 'uproot'`, or a `FileNotFoundError` on the geometry XML / `.root` data because `CALOMAPS_HOME` resolved somewhere that doesn't exist.

**Cause.** A JupyterLab kernel is launched by the notebook **server**, not by a terminal — so it inherits **nothing** you `source`d in a terminal: not the Key4hep stack that provides `uproot`/`numpy`, and not `$CALOMAPS_HOME` / `$CALOMAPS_DATA_BASE`. The generic **"Python 3 (ipykernel)"** in the picker is the server's bare `/opt/conda` Python and has none of the Key4hep packages. (Subtlety: a *sourced terminal's* `jupyter kernelspec list` shows a **different** set of kernels than the notebook picker — the picker is the server's list. Check the server's view with `/opt/conda/bin/jupyter kernelspec list`.)

**Fix.** Use the kernels whose launcher *sources the environment itself*: **`Key4hep (CPU)`** for notebooks 00–02 and **`Key4hep + GPU`** for notebook 03. `source ~/setup_calomaps.sh` registers `Key4hep (CPU)` (reload the JupyterLab browser tab if the picker doesn't list it yet); `bash setup/setup_gpu_kernel.sh` registers the GPU one. The notebooks are saved to auto-select the right kernel, so this usually only bites if you switch kernels by hand. (The notebooks also self-locate `CALOMAPS_HOME` from the kernel's working directory, so they still work even though the kernel has no `$CALOMAPS_HOME` set.)

---

## JupyterLab tab hangs / "reloads forever"

**Symptom.** The JupyterLab tab spins or reloads endlessly and never lands in the Lab UI. Reloading the same Lab URL doesn't help — it just keeps retrying. This is almost always a **hung single-user server (your spawned pod)**, not an EAF outage.

**Workaround.** Go around the dead server via the **Hub control panel**, not the Lab tab:

1. Open the Hub home directly: <https://eaf.fnal.gov/hub/home> (don't reload the Lab tab).
2. If your server shows **running** → click **Stop My Server**, wait for it to fully stop, then **Start My Server** to respawn a fresh pod. This clears a hung kernel/server in the large majority of cases.
3. Stuck on **"Spawning… / pending"** for more than ~2–3 min → backend-side (the pod can't schedule, or a CephFS `/home` PVC hiccup), not you. Wait, then retry; if it never comes up it's a facility issue.
4. Bounced to a **login** page → your SSO/CILogon session expired; re-authenticate.
5. Even `/hub/home` won't load → facility outage or your network. Recall EAF is only reachable from the Fermilab network/VPN (see "Getting started on EAF" step 1); try another network, and check the EAF status / `#eaf-users` support channel.

Nothing is lost across a restart: your clone and `$CALOMAPS_DATA_BASE` live under `/home/<user>`, a per-pod PVC that persists across restarts, and CVMFS persists too. There is no way to "kick" a hung pod from your laptop — the Hub control panel is the only lever.

---

## Training the ensembles on a CPU-only EAF session thrashes across all cores

Training the quantile ensembles without a GPU (a CPU-only EAF session, or `cuda=False`
because CVMFS ships CPU-only torch) can stall: PyTorch parallelizes each tiny full-batch
step across every core, and for a model this small (~4,400 parameters, full-batch) the
threading overhead dominates. You will see ~2000% CPU with *no* per-model progress for
many minutes. Pin it to a single thread and it runs far faster (~15-30 s per model,
~30 min for all four ensembles on CPU):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python analysis/train_ensembles.py --particle gamma
```

A GPU session is still faster end-to-end; this is the fallback when none is available.

---

## Where else to check

- **handbook.md §14** — code-level errors and project-specific gotchas (CUDA torch, scripts that wipe data dirs, etc.)
- **EAF documentation** — <https://eafjupyter.readthedocs.io/>, and `eaf-support@fnal.gov` for facility issues
- The DD4hep, Geant4, and Key4hep upstream issue trackers if you're hitting something deeper.
