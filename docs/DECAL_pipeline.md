# DECAL R&D: The Complete A-to-Z Pipeline

*Expanded Edition: From Environment Setup to Physics Reconstruction*

*(Transcribed from the project's founding design document, "DECAL R&D: Complete Simulation to Reconstruction Pipeline (Expanded)". Where details conflict, [handbook.md](handbook.md) reflects the current pipeline.)*

This document is an end-to-end guide to the Digital Calorimeter (DECAL) R&D framework. It covers everything from establishing your CVMFS environment and understanding the underlying C++ GEANT4 wrappers, to modifying the detector geometry, calibrating it using deep neural networks, and extracting fundamental physics properties.

## Part 0: The Physics (Why are we doing this?)

Traditional calorimeters measure the **Analog Energy** deposited by a particle shower. However, DECALs propose a radical alternative: **Digital Hit Counting**. Instead of measuring *how much* energy was deposited in a pixel, the readout is binary (1 if hit, 0 if not). Why throw away information? It comes down to two competing physics regimes:

- **The Low-Energy Advantage (Landau Fluctuations):** When a charged particle passes through a microscopically thin layer of silicon, energy loss follows a highly skewed Landau distribution. Rare, explosive energy dumps (Delta rays) cause massive variance in analog readouts. A binary hit-counter acts as a mathematical filter, truncating the Landau tail and dramatically improving resolution at low energies (e.g., < 20 GeV).
- **The High-Energy Catastrophe (Pixel Saturation):** At high energies (e.g., 400 GeV), the electromagnetic shower becomes incredibly dense. Dozens of secondary particles will strike the exact same silicon pixel. An analog readout measures them all; a binary readout registers a single "1" and saturates. This causes a massive, non-linear collapse in the detector's response.

## Part 1: Establishing the Simulation Environment

Before modifying any code, you must initialize the software environment. High-Energy Physics relies on a massive software stack called **Key4hep**, which contains GEANT4, ROOT, and DD4hep. Because this software is terabytes in size, it is not installed on your local machine. It is hosted on a globally distributed, read-only network drive called **CVMFS**.

### 1.1 Initializing Key4hep

Every time you open a new terminal on the Elastic Analysis Facility (EAF) or lxplus, you must attach your session to CVMFS:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
```

This single command sets up your `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH` so your terminal knows where GEANT4 and all necessary C++ compilers live.

*(Note: this project's `setup_calomaps.sh` pins the release to `-r 2026-02-01` for reproducibility, and also injects `~/lib_hack` into `LD_LIBRARY_PATH` to work around an OpenGL conflict.)*

## Part 2: Understanding the Hardware (DD4hep XML)

Writing GEANT4 detector geometry directly in C++ is incredibly tedious and error-prone. Instead, we use **DD4hep**, a toolkit that allows you to define your detector using compact XML files. When the simulation runs, DD4hep reads the XML and automatically generates the complex C++ GEANT4 geometry in the background.

### 2.1 The Baseline DECAL Edits (What we have already changed)

If you open standard calorimeter geometries, you will find coarse analog pads (e.g., 5mm × 5mm). To turn this into a DECAL testbed, **the following fundamental edits have already been made to the baseline XML:**

- **Material:** Active sensor material changed to ultra-thin Silicon to mimic Monolithic Active Pixel Sensors (MAPS).
- **Segmentation:** `grid_size` shrunk from millimeters down to **25 µm × 25 µm**. This extreme granularity is what allows individual particle tracks to be counted rather than measuring bulk energy.

*(Note: the 25 µm here — and in the snippet below — is the PDF's illustrative value. This repository's implemented pitch is **100 µm**, set by `ECal_cell_size` in `geometry/SiD_TestBeam.xml`; see notebook 00 for how to change it.)*

### 2.2 Anatomy of the Readout Block

The most critical section in `my_decal_geom.xml` (or in this project, `my_custom_ecal.xml`) is the `<readout>` block:

```xml
<readout name="ECalBarrelHits">
    <!-- 1. The Physical Pixels -->
    <segmentation type="CartesianGridXY" grid_size_x="25*um" grid_size_y="25*um"/>

    <!-- 2. The Data Encoding (Bitfield) -->
    <id>system:8,barrel:3,module:4,layer:6,slice:5,x:32:-16,y:-16</id>
</readout>
```

- **Segmentation:** Physically divides the silicon plane. *Experiment here!* Change `25*um` to `50*um` or `10*um` to see how pixel size drives the high-energy saturation catastrophe.
- **The ID String (Bitfield Encoding):** Detector saves a 3D coordinate by packing it into a single 64-bit integer. Each entry is `name:width` or `name:start:width`: `system:8` = 8 bits for system ID; `layer:6` = 6 bits for layer; `x:32:-16` = the X field *starts at bit 32* with a *signed 16-bit width* (a negative width means signed, so pixel indices can go negative either side of the module centre). You generally **do not** need to edit this string — but this is why EDM4hep ROOT files output weirdly nested branch names.

## Part 3: Executing the GEANT4 Simulation

`ddsim` is the command-line tool that acts as the steering wheel for GEANT4. The snippet below is the original illustrative recipe (electrons at a fixed energy grid, 25 um pixels). **The implemented CALOMAPS pipeline differs:** it fires **photons** (`gamma`) over a **uniform 5-400 GeV momentum spectrum** at **100 um** pixels, driven by `sim/run_sim.py` + the `CALOMAPS_GUN_*` environment variables (see notebook 00 / handbook §9). Read the loop below as conceptual:

```bash
for E in 5 10 20 50 100 200 300 400; do
    echo "Simulating ${E} GeV Electrons..."
    ddsim --compactFile my_decal_geom.xml \
          --runType run \
          --events 1000 \
          --enableG4Gun \
          --gun.particle e- \
          --gun.energy ${E}*GeV \
          --outputFile data/sim_electron_${E}GeV.root
done
```

**Command breakdown:**
- `--compactFile`: tells GEANT4 to build the geometry from your XML.
- `--enableG4Gun`: turns on the built-in particle gun.
- `--gun.particle e-`: shoots electrons. Try `gamma` or `pi+` to see how hadronic vs electromagnetic showers differ in saturation.

## Part 4: The EAF Python GPU Overlay Setup

The standard Key4hep environment defaults to CPU-only PyTorch. To train Deep Ensembles quickly, build a Python virtual environment overlay that connects to the physical GPUs on EAF nodes.

### 4.1 Build the Virtual Environment

```bash
python -m venv --system-site-packages my_gpu_env
source my_gpu_env/bin/activate
pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
python -m ipykernel install --user --name=my_gpu_env \
    --display-name="Key4hep + GPU"
```

### 4.2 Inject the JupyterHub Wrapper

Because EAF JupyterHub spawns clean, sterile kernels, it will crash if it tries to run your environment without loading CVMFS first. Create this wrapper script to force CVMFS to load before Python boots:

```bash
cd ~/.local/share/jupyter/kernels/my_gpu_env
cat << 'EOF' > wrapper.sh
#!/bin/bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
source ~/my_gpu_env/bin/activate
exec python -m ipykernel_launcher "$@"
EOF
chmod +x wrapper.sh
```

Open `kernel.json` in that folder and replace the path to the python binary with the path to your new `wrapper.sh` script.

> **Note:** this PDF recipe is superseded in this project by the one-command `setup/setup_gpu_kernel.sh` (see [handbook.md](handbook.md) §11.2); the manual recipe here is kept for reference only.

## Part 5: Software Reconstruction (Jupyter Pipeline)

**Critical workflow rule:** Switching Jupyter kernels clears your RAM. The pipeline is split into two phases:

1. **Extraction (default Key4hep kernel):** Use `uproot` to read the `.root` files, extract True Energy and Visible Signals, save with `np.savez_compressed('decal_data.npz', ...)`.
2. **Machine learning (Key4hep + GPU kernel):** Switch kernels, load the `.npz`, run the PyTorch pipeline.

### 5.1 The Forward Model (Deep Quantile Ensembles)

Train the neural network to map True Energy → Visible Energy (E_true → E_vis). Do **not** train an inverse model, because asking a neural net to guess the true energy bakes the flat training spectrum into its weights, causing massive "Prior Bias" in real experiments.

Two critical normalizations:

- **Dimensionless target:** predict Fractional Response (E_vis / E_true) so the math works whether units are GeV or raw integer hits.
- **1-Sigma band:** use Pinball Loss targeting the 15.87% and 84.13% quantiles to capture asymmetric physical tails without assuming Gaussian.

```python
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

# Convert absolute units to fractional response
y_frac = y_data / x_data
x_max, y_frac_max = np.max(x_data), np.max(y_frac)
x_norm, y_norm = x_data / x_max, y_frac / y_frac_max

# 1-sigma quantiles for Pinball Loss
quantiles = [0.1587, 0.5000, 0.8413]
```

### 5.2 Event Reconstruction (Neyman Inversion)

Use the neural network as a surrogate simulator and numerically invert it with a grid-search Neyman Construction. When a cluster curve bends backward due to saturation, strict root-finders crash. The grid-search handles this gracefully — error bounds explode to infinity rather than the code crashing.

```python
def invert_signal_grid(y_obs, e_grid, signal_curve):
    """Grid-search inversion that stays well-defined when the signal curve saturates."""
    idx = np.argmin(np.abs(signal_curve - y_obs))
    return e_grid[idx]

# Neyman Crossover: Upper True Energy limit comes from the Lower signal curve.
e_reco_high = invert_signal_grid(y_expected, e_grid, f_low_grid)
e_reco_low  = invert_signal_grid(y_expected, e_grid, f_high_grid)
resolution  = (e_reco_high - e_reco_low) / (2.0 * e_true)
```

## Part 6: Interpreting the Physics Dashboard

The pipeline outputs a 3-Panel Physics Dashboard. This is how you verify if XML hardware modifications succeeded:

| Plot Panel | What it shows | How to read it |
|---|---|---|
| **Panel 1: Reconstructed Linearity** | E_reco / E_true | Should be perfectly flat at 1.0. If it deviates: inversion algorithm failed, or detector has zero predictive power remaining. |
| **Panel 2: Reconstructed Resolution** | σ_reco / E_true | Watch the 400 GeV mark. If Hits or Clusters curves shoot violently upward: pixel size too large, catastrophic saturation. |
| **Panel 3: Stochastic Term** | σ/E vs 1/√E | In the stochastic regime the data form a straight line through the origin; slope = hardware's stochastic capability *a*. Left-side (high-energy) deviations highlight the **constant term *b*** from leakage and saturation; the noise term *c* enters as c/E and matters at the low-energy (right) side. |
