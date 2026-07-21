# Provenance: baseline_sid_o2_v03

The XML files in this directory are taken from the **SiD detector concept**, specifically the **o2_v03** compact, released October 2017.

- **Source**: SLAC SiD project (Silicon Detector Letter of Intent, ILC/LCC).
- **Original authors**: Norman Graf, Jeremy McCormick, Dan Protopopescu (per the `<info>` block in `SiD_o2_v03.xml`).
- **Confluence**: <https://confluence.slac.stanford.edu/display/ilc/sidloi>
- **Date imported into CALOMAPS**: April 2025 (file mtimes preserved from upstream as `Apr 6 2025`).
- **Modifications since import**: **None.** The files in this directory are intended to be a frozen snapshot of the upstream SiD geometry. Do not edit them. If you need to change a subdetector definition, make a copy at the `geometry/` root and modify that.

## What's actually used by CALOMAPS at runtime

The top-level compact [`../SiD_TestBeam.xml`](../SiD_TestBeam.xml) loads from this directory:

- `elements.xml`, `materials.xml` — periodic-table elements and material definitions. **Required** at runtime; CALOMAPS uses many of these materials (Silicon, TungstenDens24, Air, etc.) in the custom ECal definition.

All other `*_o2_v03.xml` files (ECalBarrel, ECalEndcap, HCal\*, SiTracker\*, SiVertex\*, MuonBarrel/Endcap, BeamCal, LumiCal, Solenoid, SteppedMuon, Support, BeamPipe) are **commented out** in `SiD_TestBeam.xml` — they are not loaded at simulation time. We keep them in the repo as:

1. Reference for what a full SiD compact looks like (in case you want to enable the tracker or HCal for a future study).
2. Evidence of what was changed vs left alone (helps with publication-style reproducibility claims).
3. Material/dimension constants that other XMLs reference are still defined here even if subdetector volumes are not built.

## What CALOMAPS replaces

The single SiD file we *don't* use is [`ECalBarrel_o2_v03.xml`](ECalBarrel_o2_v03.xml). It is replaced by [`../my_custom_ecal.xml`](../my_custom_ecal.xml), which defines the DECAL barrel:

- Same envelope dimensions as SiD ECal Barrel (rmin = 1264 mm, rmax = 1403 mm, half-length = 1765 mm, 12-fold dodecagonal symmetry).
- Same overall layer scheme for the sampling stack (20 thin + 10 thick W/Si layers) — with one
  deliberate difference: the upstream `ECalBarrel_o2_v03.xml` starts with an extra `repeat="1"`
  bare-silicon layer (Si/Cu/Kapton/Air, **no tungsten** — a preshower-style first sensor, 31 Si
  layers total). `my_custom_ecal.xml` drops it, so CALOMAPS has exactly **30** Si layers, every
  one behind an absorber. Everything downstream (layer indexing in the notebooks,
  `analysis/pixelav_converter.py`, the handbook) assumes the 30-layer scheme.
- **Different**: pixel pitch is 100 µm × 100 µm (vs SiD's millimeter-scale pads), defined by the `ECal_cell_size` constant in `../SiD_TestBeam.xml`. This is the central modification.

## How to verify the inheritance

A collaborator should be able to reproduce CALOMAPS starting from a fresh SiD checkout by:

1. Take the `o2_v03` SiD compact (elements, materials, all subdetector XMLs).
2. Copy `SiD_o2_v03.xml` to a new top-level file `SiD_TestBeam.xml`.
3. Comment out every subdetector include except the ECal barrel.
4. Replace `<include ref="ECalBarrel_o2_v03.xml"/>` with `<include ref="my_custom_ecal.xml"/>`.
5. Set `ECal_cell_size` to `0.1*mm` in `SiD_TestBeam.xml`.
6. Zero the magnetic field (`inner_field="0.0*tesla"`, `outer_field="0.0*tesla"`).

That's the entirety of the SiD → CALOMAPS transformation.
