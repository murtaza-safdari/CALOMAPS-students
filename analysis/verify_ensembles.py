"""Load trained ensembles and produce the 3-panel reconstruction dashboard.

Reads:
    $CALOMAPS_HOME/models/decal_extracted_data_<particle>.npz
    $CALOMAPS_HOME/models/<ensemble_dir>/ens_{analog,mip,hits,cluster}.pth

Writes:
    $CALOMAPS_HOME/models/figures/dashboard_linearity.png
    $CALOMAPS_HOME/models/figures/dashboard_resolution.png

`--particle {gamma,pi+}` selects the per-particle npz and the default
`saved_ensembles_gpu_<particle>` dir (override with --ensemble-dir).
"""
import sys, os, argparse

# Pick up cu121 torch if available (see analysis/train_ensembles.py)
_VENV = "/tmp/cu_torch_env/lib/python3.13/site-packages"
if os.path.isdir(_VENV):
    sys.path = [p for p in sys.path if "py-torch" not in p]
    sys.path.insert(0, _VENV)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from quantilenet import load_ensemble
from dashboard import (
    get_interpolators, reco_metrics_over_grid,
    plot_dashboard, reco_closure_events, plot_heldout_closure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle", default=os.environ.get("CALOMAPS_GUN_PARTICLE", "gamma"),
                        help="gamma or pi+ (default: gamma, or $CALOMAPS_GUN_PARTICLE)")
    parser.add_argument("--ensemble-dir", default=None,
                        help="Subdir under $CALOMAPS_HOME/models/ holding ens_*.pth files "
                             "(default: saved_ensembles_gpu_<particle>)")
    parser.add_argument("--show", action="store_true",
                        help="Also call plt.show() (useful when running inside a notebook).")
    args = parser.parse_args()
    tag = "gamma" if args.particle == "gamma" else args.particle.replace("+", "plus").replace("-", "minus")

    calomaps_home = os.environ.get("CALOMAPS_HOME", os.path.expanduser("~/CALOMAPS"))
    # Accept the canonical per-particle names first (what notebooks 02/03 write here), then
    # legacy / alternate names so instructor-provided artifacts also load (unsuffixed photon
    # npz; saved_ensembles_gpu_v2/ and saved_ensembles_piplus/ ensemble dirs).
    npz_cands = [os.path.join(calomaps_home, "models", f"decal_extracted_data_{tag}.npz")]
    if tag == "gamma":
        npz_cands.append(os.path.join(calomaps_home, "models", "decal_extracted_data.npz"))
    npz_path = next((p for p in npz_cands if os.path.exists(p)), None)
    if npz_path is None:
        sys.exit("ERROR: no extracted-data npz found. Tried:\n  " + "\n  ".join(npz_cands)
                 + "\nRun notebooks/02_data_extraction.ipynb first (set the particle env vars for pions).")
    if args.ensemble_dir:
        dir_cands = [os.path.join(calomaps_home, "models", args.ensemble_dir)]
    else:
        dir_cands = [os.path.join(calomaps_home, "models", f"saved_ensembles_gpu_{tag}"),
                     os.path.join(calomaps_home, "models",
                                  "saved_ensembles_gpu_v2" if tag == "gamma" else f"saved_ensembles_{tag}")]
    ens_dir = next((d for d in dir_cands if os.path.isdir(d)), None)
    if ens_dir is None:
        sys.exit("ERROR: no ensemble dir found. Tried:\n  " + "\n  ".join(dir_cands)
                 + "\nTrain first (notebook 03 or analysis/train_ensembles.py), or pass --ensemble-dir.")
    fig_dir = os.path.join(calomaps_home, "models", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"torch:  {torch.__file__}  v{torch.__version__}  cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"data:   {npz_path}")
    print(f"models: {ens_dir}")
    print(f"figures: {fig_dir}")
    print()

    # ---- load data + valid mask ----------------------------------------------
    data = np.load(npz_path)
    all_truth   = data["all_truth"]
    all_visible = data["all_visible"]
    all_mip     = data["all_mip"]
    all_hits    = data["all_hits"]
    all_cluster = data["all_cluster"]
    valid = (all_hits > 0) & (all_truth > 0) & (all_visible > 0) & (all_mip > 0) & (all_cluster > 0)
    print(f"valid events: {valid.sum()}")

    # ---- load all four ensembles ---------------------------------------------
    print("loading ensembles...")
    ens_a, xa, ya = load_ensemble(os.path.join(ens_dir, "ens_analog.pth"), device)
    ens_m, xm, ym = load_ensemble(os.path.join(ens_dir, "ens_mip.pth"),     device)
    ens_h, xh, yh = load_ensemble(os.path.join(ens_dir, "ens_hits.pth"),    device)
    ens_c, xc, yc = load_ensemble(os.path.join(ens_dir, "ens_cluster.pth"), device)
    print(f"  loaded 4 ensembles, {len(ens_a)} models each")

    # ---- build interpolators + Neyman reconstruction over a grid -------------
    print("computing dashboard metrics...")
    fla, fma, fha = get_interpolators(ens_a, xa, ya, device)
    flm, fmm, fhm = get_interpolators(ens_m, xm, ym, device)
    flh, fmh, fhh = get_interpolators(ens_h, xh, yh, device)
    flc, fmc, fhc = get_interpolators(ens_c, xc, yc, device)

    reco = {
        "Analog":  reco_metrics_over_grid(fla, fma, fha),
        "MIP":     reco_metrics_over_grid(flm, fmm, fhm),
        "Hits":    reco_metrics_over_grid(flh, fmh, fhh),
        "Cluster": reco_metrics_over_grid(flc, fmc, fhc),
    }

    # ---- dashboard plots -----------------------------------------------------
    out_prefix = os.path.join(fig_dir, "dashboard")
    plot_dashboard(reco, out_path_prefix=out_prefix, show=args.show)
    print(f"saved {out_prefix}_linearity.png and {out_prefix}_resolution.png")

    # ---- held-out event-based closure (the honest test: events no model trained on) ----
    hp = os.path.join(ens_dir, "heldout_test.npz")
    if os.path.exists(hp):
        # The split is only honest if these ensembles were trained together with it.
        # Ensembles retrained afterwards into this dir (e.g. interactively via nb03,
        # which trains on ALL valid events) may have trained on these very events.
        newest_ens = max(os.path.getmtime(os.path.join(ens_dir, f"ens_{k}.pth"))
                         for k in ("analog", "mip", "hits", "cluster"))
        if newest_ens > os.path.getmtime(hp) + 1.0:
            print("WARNING: the ens_*.pth files are newer than heldout_test.npz -- the "
                  "ensembles were retrained after this split was saved and may have "
                  "trained on these events, so the closure below is NOT guaranteed "
                  "honest. Rerun train_ensembles.py to refresh both together.")
        h = np.load(hp)
        heldout = {
            "Analog":  reco_closure_events(fma, h["all_visible"], h["all_truth"], label="Analog"),
            "MIP":     reco_closure_events(fmm, h["all_mip"],     h["all_truth"], label="MIP"),
            "Hits":    reco_closure_events(fmh, h["all_hits"],    h["all_truth"], label="Hits"),
            "Cluster": reco_closure_events(fmc, h["all_cluster"], h["all_truth"], label="Cluster"),
        }
        plot_heldout_closure(heldout, out_path_prefix=out_prefix, show=args.show)
        print(f"saved {out_prefix}_heldout.png  ({len(h['all_truth'])} held-out events)")
        print("\n=== held-out closure  (response E_reco/E_true | resolution std, "
              "@ actual bin centre) ===")
        for energy in (10, 100, 300):
            row = f"  E~{energy:>3d} GeV: "
            for k in ("Analog", "MIP", "Hits", "Cluster"):
                et, resp, res = heldout[k]
                if len(et):
                    i = int(np.argmin(np.abs(et - energy)))
                    row += f" {k}={resp[i]:.3f}/{res[i]:.3f}@{et[i]:.0f}"
            print(row)
    else:
        print("(no heldout_test.npz in the ensemble dir -- retrain with the updated "
              "train_ensembles.py to enable the held-out closure)")

    # ---- headline numbers ----------------------------------------------------
    print()
    print("=== headline resolutions (sigma_reco / E_true) ===")
    et = reco["Analog"][0]
    for energy in (10, 100, 300):
        idx = np.argmin(np.abs(et - energy))
        a = reco["Analog"][2][idx]
        m = reco["MIP"][2][idx]
        h = reco["Hits"][2][idx]
        c = reco["Cluster"][2][idx]
        print(f"  E={energy:>3d} GeV:  analog={a:.4f}  mip={m:.4f}  hits={h:.4f}  cluster={c:.4f}")


if __name__ == "__main__":
    main()
