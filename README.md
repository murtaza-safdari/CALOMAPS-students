# CALOMAPS — Ultra-High Granularity Digital Calorimeter (DECAL) R&D

A research framework for digital electromagnetic calorimeter (DECAL) studies using simulated Monolithic Active Pixel Sensor (MAPS) silicon layers. CALOMAPS pairs a DD4hep / Geant4 simulation of a custom 100 µm-pitch silicon-tungsten ECal barrel with a PyTorch Deep Quantile Ensemble surrogate model and a Neyman-construction energy reconstruction, producing the standard 3-panel calorimeter performance dashboard (linearity, resolution, stochastic term).

The pipeline runs end-to-end on the [Fermilab Elastic Analysis Facility (EAF)](https://eaf.fnal.gov). Notebook 00 generates your data; the analysis notebooks (`01 → 02 → 03`) ship as **scaffolds**: the context, plumbing, and plotting are given; the science cells are left blank, each stating a task, a hint, and a sanity check on the output. Reference solutions are not distributed with this repository. The clustering study at the end of notebook 03 is open — no reference solution exists, because the answer isn't known.

---

## Why ultra-high granularity?

Traditional electromagnetic calorimeters measure analog energy in millimeter-scale cells. A DECAL inverts this: cell sizes are tens of micrometers (here, 100 µm × 100 µm) and the readout is **binary** (hit / no-hit). The bet is that at low shower energy, binary readout truncates the Landau tail of energy deposits and improves resolution; at high energy, pixel saturation makes the digital readout lose information that an analog readout would have kept. CALOMAPS quantifies this trade-off and provides a framework for pitch / particle-type / geometry scans.

See [docs/DECAL_pipeline.md](docs/DECAL_pipeline.md) for the full physics writeup and [docs/handbook.md](docs/handbook.md) for the operational pipeline.

---

## Getting started on EAF (from zero)

Everything runs in a browser JupyterLab session on EAF — no software is installed on your own machine.

1. **Accounts.** A Fermilab computing account is required (set up with your institutional onboarding); EAF access comes with it. <https://eaf.fnal.gov> is reachable from the Fermilab network (on-site WiFi) — if the page won't load off-site, that's why. Details: [docs/handbook.md](docs/handbook.md) §6.1.

2. **Spawn a server.** Log in at <https://eaf.fnal.gov> and pick the **"GPU A100 10GB"** spawner profile (or the current GPU non-CMS profile). Everything runs on this one server; the GPU itself is only used by notebook 03. After ~30 s you land in JupyterLab.

3. **Open a terminal.** File → New → Terminal.

4. **Clone the repository** (into a folder named `CALOMAPS`):

   ```bash
   cd ~
   git clone https://github.com/murtaza-safdari/CALOMAPS-students.git CALOMAPS
   ln -s ~/CALOMAPS/setup/setup_calomaps.sh ~/setup_calomaps.sh
   ```

   Clone wherever you like — `$HOME` is simplest and always available. `setup_calomaps.sh` locates the repo from its own path, so nothing needs to match a fixed location.

5. **Load the environment** (in every new terminal):

   ```bash
   source ~/setup_calomaps.sh
   ```

   `source` runs the script inside your current shell so the variables it sets persist. The script loads the Key4hep software stack (Geant4, ROOT, DD4hep, Python libraries) from CVMFS — a read-only network filesystem that streams pre-installed software on demand, so nothing lands in your home quota — and sets `$CALOMAPS_HOME` (the repo, located automatically) and `$CALOMAPS_DATA_BASE` (where simulation output goes). It is safe to source repeatedly. Confirm the environment with the smoke test in [docs/handbook.md](docs/handbook.md) §8.

   > **Disk note:** simulation data goes to `$CALOMAPS_DATA_BASE` (defaults to `~/CALOMAPS-data` on your `/home`, ~23 GB quota). That quota is all you have for now, so keep datasets modest and delete runs you no longer need.

6. **Kernels.** Notebooks 00–02 use the **`Key4hep (CPU)`** kernel, which `setup_calomaps.sh` registered for you in step 5 (reload the JupyterLab browser tab if the picker doesn't list it yet). Don't pick the generic "Python 3 (ipykernel)" — that base Python has no `uproot`. Notebook 03 needs the **`Key4hep + GPU`** kernel: register it once with `bash $CALOMAPS_HOME/setup/setup_gpu_kernel.sh` (~5–10 min — downloads ~4.4 GB of CUDA torch, longer on a busy EAF; re-run if the server restarts). Details: handbook §6.4 and §11.2.

The full setup reference — accounts, spawner profiles, the storage map, and the GPU kernel — is [docs/handbook.md](docs/handbook.md) §6.

---

## The notebooks

Run them in order; each consumes what the previous one produced.

| Notebook | What it does |
|---|---|
| [`00_simulate_your_samples`](notebooks/00_simulate_your_samples.ipynb) | Generate simulation data with `ddsim` (given in full — no blanks). A fresh clone has **no data**; run this first. |
| [`01_detector_and_data`](notebooks/01_detector_and_data.ipynb) | Parse the geometry, open one event, image a single shower three ways, measure the sampling fraction. |
| [`02_data_extraction`](notebooks/02_data_extraction.ipynb) | Reduce every event to four digital readouts (analog, MIP, hits, clusters) and save one `.npz`. |
| [`03_ml_training_and_eval`](notebooks/03_ml_training_and_eval.ipynb) | Train a Deep Quantile Ensemble per readout, invert via a Neyman construction, render the dashboard — then the open clustering study. |

Cells marked `# TODO` raise `NotImplementedError` until implemented; everything else is given. Generating a starter dataset is one command (notebook 00 documents it):

```bash
CALOMAPS_NJOBS=40 bash $CALOMAPS_HOME/sim/generate_batched.sh
```

---

## One pipeline, two particles

The notebooks run on one particle at a time, selected by a single `PARTICLE` variable near the top of each notebook (`"gamma"`, the default, or `"pi+"`; for data generation it's the `CALOMAPS_GUN_PARTICLE` env var). Dataset directories, file globs, output `.npz` names, and trained-model folders are all derived from it, so runs on different particles never clobber each other.

Photons shower electromagnetically; charged pions shower hadronically. How that difference shows up in this detector is a central thread of the analysis. Three places to compare, once both particles have been run:

1. **Shower shape** (end of nb01) — compare the densest-layer image, the per-layer slices, and the longitudinal profile. Which particle's shower is wider? Deeper? Do all layers always light up?
2. **Sampling fraction & readouts** (end of nb02) — are the readout↔energy relations equally tight? Do the longitudinal profiles peak at the same depth?
3. **Reconstruction resolution** (end of nb03) — which particle reconstructs better? Is the ranking of readouts the same?

A small table — best baseline σ/E, improved-cluster σ/E, and the relative gain, for both particles — captures the result.

---

## The clustering study (notebook 03, open)

Everything up to that point uses the baseline clusterer from nb02: `naive_clusters()` counts 8-connected pixel groups per layer, independently, and sums them. At 100 µm pitch a shower is resolved into thousands of pixels across 30 layers, and the baseline throws most of that structure away — it never looks across layers, never uses deposited charge, never splits merged cores.

The task: replace it with something better, re-extract the cluster readout, retrain just the cluster ensemble, and measure whether the resolution improved. Directions worth trying: 3-D connected components across layers; charge/energy weighting; density-based (Molière-aware) core splitting; a small learned clusterer.

The success criterion is a measured change: the baseline and improved σ/E curves side by side on the dashboard, with the difference stated as a number (e.g. "σ/E at 100 GeV: X → Y, a Z % relative change"). A change that does **not** improve the resolution is also a reportable result, together with the reason.

---

## Pipeline at a glance

```
geometry/SiD_TestBeam.xml + my_custom_ecal.xml
            │
            ▼
sim/run_sim.py  ──>  ddsim  ──>  ROOT files (sim_<particle>_part*.root)
                                      │
                                      ▼
                       notebooks/02_data_extraction.ipynb
                                      │
                                      ▼
                models/decal_extracted_data_<particle>.npz
                                      │
                                      ▼
                       analysis/quantilenet.train_one_ensemble
                                      │
                                      ▼
                models/saved_ensembles_gpu_<particle>/  (4 ensembles, 20 models each)
                                      │
                                      ▼
                      analysis/dashboard.plot_dashboard
                                      │
                                      ▼
                       3-panel physics dashboard
```

---

## Repository layout

```
CALOMAPS/
├── README.md                          ← you are here
├── docs/
│   ├── handbook.md                    ← OPERATIONAL guide: setup, simulation, training, dashboard
│   ├── troubleshooting.md             ← environment quirks and their fixes
│   └── DECAL_pipeline.md              ← PHYSICS writeup
├── setup/
│   ├── setup_calomaps.sh              ← source from a JupyterLab terminal to bootstrap the env
│   └── setup_gpu_kernel.sh            ← one-shot: register the GPU kernel for notebook 03
├── geometry/                          ← DD4hep XML detector descriptions
│   ├── SiD_TestBeam.xml               ← top-level compact passed to ddsim
│   ├── my_custom_ecal.xml             ← the DECAL barrel definition
│   └── baseline_sid_o2_v03/           ← upstream SiD baseline XMLs (untouched, for reference)
├── sim/                               ← Geant4 / DD4hep driver scripts
│   ├── run_sim.py                     ← ddsim steering (particle gun via CALOMAPS_GUN_* env vars)
│   ├── generate_batched.sh            ← parallel dataset generation (CALOMAPS_NJOBS / _NEVENTS)
│   └── generate_dataset.sh
├── analysis/                          ← Python ML / reconstruction utilities
│   ├── quantilenet.py                 ← QuantileNet model + training + load/save
│   ├── dashboard.py                   ← Neyman inversion + 3-panel dashboard plotting
│   ├── train_ensembles.py             ← CLI entry point (headless training)
│   └── verify_ensembles.py            ← CLI entry point (regenerate dashboard)
├── notebooks/                         ← the workflow: 00 → 01 → 02 → 03
└── models/                            ← created by the notebooks (gitignored):
                                          decal_extracted_data_<particle>.npz
                                          + saved_ensembles_gpu_<particle>/
```

Simulation ROOT files live **outside the repo** at `$CALOMAPS_DATA_BASE` (defaults to `~/CALOMAPS-data/`); they are produced by `sim/generate_batched.sh` and consumed by notebook 02.

---

## When something breaks

[docs/troubleshooting.md](docs/troubleshooting.md) catalogs the environment's known quirks and fixes — check it first. Code-level gotchas are in [docs/handbook.md](docs/handbook.md) §14, and §15 lists where to ask for help.

---

## Provenance

The detector geometry is built on top of the [SiD detector concept](https://confluence.slac.stanford.edu/display/ilc/sidloi) (Silicon Detector — Letter of Intent), specifically the `o2_v03` compact released October 2017 by Norman Graf, Jeremy McCormick, and Dan Protopopescu. CALOMAPS retains the SiD coordinate system, dimensions, and material definitions, but disables every subdetector except the ECal barrel (which is replaced by the custom DECAL definition in `geometry/my_custom_ecal.xml`). See [geometry/baseline_sid_o2_v03/PROVENANCE.md](geometry/baseline_sid_o2_v03/PROVENANCE.md) for the full inheritance documentation.

The deep quantile ensemble approach follows the standard pinball-loss regression pattern; the Neyman-construction inversion is implemented with `scipy.optimize.brentq`, extending the search bracket into the saturation regime and reporting points the readout can never reach as NaN (dropped) rather than clipping them to a fabricated value.

---

## Acknowledgements

This work uses [Key4hep](https://key4hep.github.io/), [DD4hep](https://dd4hep.web.cern.ch/), [Geant4](https://geant4.web.cern.ch/), [uproot](https://uproot.readthedocs.io/), and [PyTorch](https://pytorch.org/). The Elastic Analysis Facility ([EAF](https://eaf.fnal.gov)) at Fermilab provides the GPU compute and software environment.

---

## License

[MIT](LICENSE). See `LICENSE` for the full text.
