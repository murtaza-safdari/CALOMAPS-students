#!/usr/bin/env python3
"""Per-energy readout extraction for the conventional-resolution notebook
(03c_conventional_resolution.ipynb).

Runs over ONE mono-energetic ddsim dataset (a fixed-energy beam) and writes a
per-energy .npz with the four readouts that 03c consumes:
    all_truth, all_visible, all_mip, all_hits, all_cluster, E_nominal

This is the batch/per-energy CLI form of the notebook-02 extraction: it applies the +y
wedge / half-MIP cuts (given) and calls YOUR four-readout code (the compute_readouts TODO
below -- the same V/M/H/C you fill in notebook 02) on every event of a fixed-energy dataset.
Loop it over your fixed-energy datasets:

    for E in 1 2 5 10 20 50 100 200 400; do
      python analysis/extract_readouts.py \\
        --datadir $CALOMAPS_DATA_BASE/mono_gamma_${E}GeV \\
        --glob 'sim_photons_part*.root' --energy $E \\
        --out $CALOMAPS_HOME/models/mono_gamma/decal_mono_gamma_E$(printf '%04d' $E)GeV.npz
    done

Geometry sweeps: the pixel pitch (ECal_cell_size) and the layer depths are
re-read from <home>/geometry/*.xml, so point --home at the (possibly
copied-and-edited) geometry you simulated with and those follow automatically.
The barrel envelope is assumed unchanged: the wedge cut uses the baseline
NSIDES/RMIN/RMAX constants below, so if you edit ECalBarrel_rmin/rmax/symmetry,
update those constants to match. The MIP scale cannot come from geometry (it is
a measured Landau MPV, 85 keV for 320 um silicon): pass --mip-energy when you
change the silicon thickness. A startup probe cross-checks the first file's hit
lattice and layer depths against --home and aborts on a mismatch.
"""
import os, glob, re, argparse, xml.etree.ElementTree as ET
import numpy as np, uproot
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- readout constants (baseline defaults; match notebooks/02_data_extraction.ipynb).
#      main() re-resolves CELL_SIZE from --home's geometry XML and MIP_ENERGY from
#      --mip-energy, so the CLI cannot silently disagree with the simulated detector. ----
CELL_SIZE    = 0.1           # mm, 100 um pitch (ECal_cell_size; re-read per --home)
MIP_ENERGY   = 85e-6         # GeV, Landau MPV for 320 um Si (see --mip-energy)
THRESHOLD    = 0.5 * MIP_ENERGY
NSIDES       = 12              # ECalBarrel_symmetry (NOT re-read; assumed unchanged in sweeps)
SEG_HALF_DEG = 180.0 / NSIDES  # 15 deg
RMIN, RMAX   = 1264.0, 1403.0  # mm, ECalBarrel_rmin/rmax (NOT re-read; assumed unchanged)


_UNITS_MM = {'m': 1000.0, 'cm': 10.0, 'mm': 1.0, 'um': 1e-3, 'micron': 1e-3, 'nm': 1e-6}


def _pv(v, c):
    if not v: return 0.0
    if v in c: return _pv(c[v], c)
    # Whole-word unit substitution: '50*um' parses; '50um' (juxtaposition, which
    # ddsim also rejects) or an unknown unit fails loudly instead of yielding 0.
    s = re.sub(r'\b[A-Za-z_]\w*',
               lambda m: repr(_UNITS_MM[m.group(0)]) if m.group(0) in _UNITS_MM else m.group(0), v)
    try: return float(eval(s, {"__builtins__": None}, {}))
    except Exception:
        try: return float(v)
        except Exception:
            raise ValueError(f"cannot parse geometry value {v!r} "
                             f"(units understood: {', '.join(sorted(_UNITS_MM))})")


def layer_radii(calomaps_home):
    g = os.path.join(calomaps_home, "geometry")
    consts = {x.get("name"): x.get("value")
              for x in ET.parse(os.path.join(g, "SiD_TestBeam.xml")).getroot().findall(".//constant")}
    det = ET.parse(os.path.join(g, "my_custom_ecal.xml")).getroot().find(".//detector[@name='ECalBarrel']")
    # The ECalBarrel_o2_v03 driver starts the stack at rmin + ecal_barrel_tolerance (=env_safety,
    # 0.1 mm), not at rmin. The offset is uniform, so nearest-centre layer BINNING below is
    # unaffected; we add it so absolute depths are correct.
    cur, planes = _pv(det.find("dimensions").get("rmin"), consts) + 0.1, []
    for layer in det.findall("layer"):
        rep = int(layer.get("repeat", 1)); sl = layer.findall("slice")
        thick = sum(_pv(s.get("thickness"), consts) for s in sl)
        off = sioff = 0.0
        for s in sl:
            t = _pv(s.get("thickness"), consts)
            if s.get("material") == "Silicon": sioff = off + t / 2
            off += t
        for _ in range(rep):
            planes.append(cur + sioff); cur += thick
    return np.array(planes)


def cell_size_mm(calomaps_home):
    """Pixel pitch [mm] from the compact's ECal_cell_size -- the same constant that
    drives the CartesianGridXY segmentation, so the cluster readout is built on the
    simulated pitch instead of trusting a hand-synced copy."""
    g = os.path.join(calomaps_home, "geometry")
    consts = {x.get("name"): x.get("value")
              for x in ET.parse(os.path.join(g, "SiD_TestBeam.xml")).getroot().findall(".//constant")}
    v = _pv(consts.get("ECal_cell_size"), consts)
    if v <= 0:
        raise SystemExit(f"could not resolve ECal_cell_size from {g}/SiD_TestBeam.xml "
                         f"(got {v!r}) -- does --home point at the geometry you simulated with?")
    return v


def naive_clusters(x, z, layer_idx, e):
    """8-connected components per layer, summed (identical to nb02's baseline)."""
    m = e > THRESHOLD
    if not m.any():
        return 0
    xi = np.round(x[m] / CELL_SIZE).astype(np.int64)
    zi = np.round(z[m] / CELL_SIZE).astype(np.int64)
    li = layer_idx[m]
    total = 0
    for ly in np.unique(li):
        sel = li == ly
        cells_ = set(zip(xi[sel].tolist(), zi[sel].tolist()))
        seen = set()
        for c0 in cells_:
            if c0 in seen:
                continue
            total += 1
            stack = [c0]
            while stack:
                ux, uz = stack.pop()
                if (ux, uz) in seen:
                    continue
                seen.add((ux, uz))
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        v = (ux + dx, uz + dz)
                        if v in cells_ and v not in seen:
                            stack.append(v)
    return total


def compute_readouts(x, y, z, e, radii):
    """The four readouts for ONE event's hit arrays. Pure function (no ROOT) so
    it is unit-testable. Returns (V, M, H, C) or None if no hits survive the wedge.
    This is where you port nb02's V/M/H/C TODO block."""
    if len(e) == 0:
        return None
    ang = np.degrees(np.arctan2(x, y))
    seg = (np.abs(ang) < SEG_HALF_DEG) & (y > RMIN - 4) & (y < RMAX + 14)
    x, z, e, y = x[seg], z[seg], e[seg], y[seg]
    if len(e) == 0:
        return None
    layer_idx = np.argmin(np.abs(radii[None, :] - y[:, None]), axis=1)
    thr = e > THRESHOLD
    # ============================================================
    # TODO: compute the four readouts for this event's wedge hits -- the SAME four
    # you implemented in notebook 02. After the wedge cut above you have, for the
    # hits in this event:
    #   e         : pixel energies [GeV]
    #   thr       : boolean mask, True where e > THRESHOLD (1/2 MIP)
    #   x, z      : pixel coordinates [mm]   ;   layer_idx : silicon layer per hit
    #     V (analog) : sum of ALL hit energies in the segment                    -> float
    #     M (MIP)    : over FIRED pixels, sum of max(1, round(E_pix/MIP_ENERGY)) -> float
    #     H (hits)   : number of fired pixels (count of the thr mask)            -> int
    #     C (cluster): naive_clusters(x, z, layer_idx, e)   (given above)        -> int
    # Solution: not distributed with this repository.
    # ============================================================
    raise NotImplementedError("port your notebook-02 readout code (V, M, H, C) here")
    return V, M, H, C


def process_single_file(filepath, radii):
    br = ["ECalBarrelHits.position.x", "ECalBarrelHits.position.y",
          "ECalBarrelHits.position.z", "ECalBarrelHits.energy",
          "MCParticles.momentum.x", "MCParticles.momentum.y",
          "MCParticles.momentum.z", "MCParticles.mass"]
    T, V, M, H, C = [], [], [], [], []
    try:
        with uproot.open(filepath) as f:
            tr = f["events"]
            if tr.num_entries == 0:
                return None
            a = tr.arrays(br)
            truth = np.sqrt(a["MCParticles.momentum.x"][:, 0]**2 +
                            a["MCParticles.momentum.y"][:, 0]**2 +
                            a["MCParticles.momentum.z"][:, 0]**2 +
                            a["MCParticles.mass"][:, 0]**2)
            hx, hy = a["ECalBarrelHits.position.x"], a["ECalBarrelHits.position.y"]
            hz, he = a["ECalBarrelHits.position.z"], a["ECalBarrelHits.energy"]
            for ev in range(len(he)):
                out = compute_readouts(np.asarray(hx[ev]), np.asarray(hy[ev]),
                                       np.asarray(hz[ev]), np.asarray(he[ev]), radii)
                if out is None:
                    continue
                v, m, h, c = out
                T.append(float(truth[ev])); V.append(v); M.append(m); H.append(h); C.append(c)
    except Exception as ex:
        print(f"  failed {os.path.basename(filepath)}: {ex}")
        return None
    return T, V, M, H, C


def _init_readout_constants(cell_size, mip_energy):
    """Executor initializer: propagate the resolved pitch/MIP scale into each worker
    process (module globals would only survive fork, not spawn)."""
    global CELL_SIZE, MIP_ENERGY, THRESHOLD
    CELL_SIZE, MIP_ENERGY, THRESHOLD = cell_size, mip_energy, 0.5 * mip_energy


def _lattice_probe(x, z, y, radii):
    """Return an error message if the hits are inconsistent with the resolved
    geometry (i.e. --home does not match the simulated detector), else None.
    Fraction/mean based: hits from the wedge's rotated neighbour staves sit off
    the +y module's lattice, so a small off-lattice tail is normal."""
    frac_off = max(np.mean(np.abs(x / CELL_SIZE - np.round(x / CELL_SIZE)) > 0.05),
                   np.mean(np.abs(z / CELL_SIZE - np.round(z / CELL_SIZE)) > 0.05))
    if frac_off > 0.10:
        return (f"{frac_off:.0%} of hit positions are off the {CELL_SIZE} mm pixel "
                f"lattice -- the simulated pitch is finer than {CELL_SIZE} mm (or unrelated)")
    odd_frac = np.mean(np.abs(x / (2 * CELL_SIZE) - np.round(x / (2 * CELL_SIZE))) > 0.25)
    if odd_frac < 0.05:
        return (f"hits only populate every second {CELL_SIZE} mm cell -- the "
                f"simulated pitch is coarser than {CELL_SIZE} mm")
    mean_dy = float(np.mean(np.min(np.abs(y[:, None] - radii[None, :]), axis=1)))
    if mean_dy > 0.5:
        return (f"hit depths sit {mean_dy:.2f} mm (mean) from the nearest layer "
                f"plane -- the layer stack (Si thickness) does not match")
    return None


def _probe_geometry(files, radii, min_hits=500):
    """Read hits from the first readable file and abort if they are not on the
    pixel lattice / layer planes implied by --home -- catches extracting a swept
    dataset with the wrong geometry BEFORE burning the whole pool run on it."""
    xs, ys, zs, n = [], [], [], 0
    for fp in files:
        try:
            with uproot.open(fp) as f:
                a = f["events"].arrays(["ECalBarrelHits.position.x",
                                        "ECalBarrelHits.position.y",
                                        "ECalBarrelHits.position.z"])
        except Exception:
            continue
        for ev in range(len(a["ECalBarrelHits.position.x"])):
            hx = np.asarray(a["ECalBarrelHits.position.x"][ev], float)
            hy = np.asarray(a["ECalBarrelHits.position.y"][ev], float)
            hz = np.asarray(a["ECalBarrelHits.position.z"][ev], float)
            ang = np.degrees(np.arctan2(hx, hy))
            m = (np.abs(ang) < SEG_HALF_DEG) & (hy > RMIN - 4) & (hy < RMAX + 14)
            xs.append(hx[m]); ys.append(hy[m]); zs.append(hz[m]); n += int(m.sum())
            if n >= min_hits:
                break
        break                          # first readable file is enough
    if n < min_hits:
        return                         # too few hits to judge -- proceed
    err = _lattice_probe(np.concatenate(xs), np.concatenate(zs), np.concatenate(ys), radii)
    if err:
        raise SystemExit(f"geometry mismatch: {err}. Is --home pointing at the "
                         f"geometry these files were simulated with?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True, help="dir of mono-energetic .root files")
    ap.add_argument("--glob", default="sim_photons_part*.root")
    ap.add_argument("--energy", type=float, required=True, help="nominal beam energy [GeV]")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--home", default=os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS")),
                    help="config root whose geometry/*.xml supplies the pixel pitch and layer "
                         "depths (default: $CALOMAPS_HOME, else ~/CALOMAPS); for sweeps, point "
                         "at the copied-and-edited geometry you simulated with")
    ap.add_argument("--mip-energy", type=float, default=MIP_ENERGY, metavar="GEV",
                    help="MIP Landau MPV in GeV (default 85e-6 = 320 um Si). Re-derive and pass "
                         "this when the Si thickness changes; the 1/2-MIP threshold follows it.")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    try:
        radii = layer_radii(args.home)
    except FileNotFoundError:
        raise SystemExit(f"--home {args.home!r} has no geometry/SiD_TestBeam.xml + "
                         f"my_custom_ecal.xml -- point it at the repo/config root")
    _init_readout_constants(cell_size_mm(args.home), args.mip_energy)
    files = sorted(glob.glob(os.path.join(args.datadir, args.glob)))
    if not files:
        raise SystemExit(f"no files match {os.path.join(args.datadir, args.glob)}")
    print(f"E={args.energy} GeV: {len(files)} files, {args.workers} workers  "
          f"[pitch {CELL_SIZE*1e3:g} um from --home geometry, MIP {MIP_ENERGY*1e6:g} keV]")
    _probe_geometry(files, radii)      # abort early on a data/geometry mismatch

    T, V, M, H, C = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_readout_constants,
                             initargs=(CELL_SIZE, MIP_ENERGY)) as ex:
        futs = {ex.submit(process_single_file, f, radii): f for f in files}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                T += r[0]; V += r[1]; M += r[2]; H += r[3]; C += r[4]
            if n % 50 == 0:
                print(f"  {n}/{len(files)}")

    all_truth = np.array(T)
    if all_truth.size == 0:
        raise SystemExit("0 events extracted -- check dataset/glob")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out,
                        all_truth=all_truth,
                        all_visible=np.array(V, np.float32),
                        all_mip=np.array(M, np.float32),
                        all_hits=np.array(H, np.int64),
                        all_cluster=np.array(C, np.int64),
                        E_nominal=np.float64(args.energy))
    print(f"  {all_truth.size} events -> {args.out}  "
          f"(truth mean={all_truth.mean():.2f} std={all_truth.std():.3f} GeV)")


if __name__ == "__main__":
    main()
