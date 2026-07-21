"""Plotting + reconstruction utilities for the DECAL 3-panel physics dashboard.

Given trained `QuantileNet` ensembles, this module:
  1. builds SciPy interpolators of the ensemble-averaged quantile curves vs true
     energy, with quantile monotonicity enforced (`get_interpolators`),
  2. runs Neyman-construction reconstruction over a test energy set
     (`reco_metrics_over_grid`),
  3. produces the 3-panel dashboard: a median self-inversion *self-consistency*
     check, reconstructed resolution vs E, and the stochastic resolution vs
     1/sqrt(E) with a fitted a/sqrt(E) (+) b (+) c/E term (`plot_dashboard`).

Used by analysis/verify_ensembles.py and the 03_ml_training_and_eval notebook.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit


# ---- Neyman-inversion reconstruction -----------------------------------------

def get_interpolators(ensemble, x_max: float, y_frac_max: float,
                      device: torch.device,
                      e_grid: np.ndarray = None):
    """Build (f_low, f_med, f_high) SciPy interpolators of the ensemble-averaged
    quantile curves vs true energy. Used as input to Brent inversion below.

    The three quantile heads are trained independently, so in sparse/wide regions
    they can CROSS (q_low > q_med, etc.). We sort the three curves at each energy
    so the band edges are monotone by construction (q_low <= q_med <= q_high),
    which keeps the inverted resolution well-defined (non-negative).
    """
    if e_grid is None:
        e_grid = np.linspace(1, 500, 1000)
    x_tensor = torch.tensor(e_grid / x_max, dtype=torch.float32,
                            device=device).unsqueeze(1)
    preds = []
    for m in ensemble:
        m.eval()
        with torch.no_grad():
            preds.append(m(x_tensor).cpu().numpy())
    avg = np.mean(preds, axis=0)
    preds_abs = avg * y_frac_max * e_grid[:, None]
    preds_abs = np.sort(preds_abs, axis=1)      # enforce quantile monotonicity (anti-crossing)
    f_low = interp1d(e_grid, preds_abs[:, 0], kind="linear", fill_value="extrapolate")
    f_med = interp1d(e_grid, preds_abs[:, 1], kind="linear", fill_value="extrapolate")
    f_high = interp1d(e_grid, preds_abs[:, 2], kind="linear", fill_value="extrapolate")
    return f_low, f_med, f_high


def invert_brent(y_obs: float, f_curve, lo: float = 5.0, hi: float = 450.0,
                 hi_max: float = 5000.0) -> float:
    """Find E such that f_curve(E) == y_obs.

    Brent's method needs the objective to change sign between `lo` and `hi`. When
    the (increasing) readout response saturates, the requested `y_obs` may sit
    above f_curve(hi); we EXTEND the upper bracket geometrically up to `hi_max`
    before giving up. If `y_obs` lies below f_curve(lo) we clip to `lo` (physical
    floor). If no bracket exists even at `hi_max` -- genuine saturation, where the
    readout can never reach `y_obs` -- we return NaN so the caller drops the point
    rather than silently clipping to a fabricated value (which previously turned
    the saturation regime into an artificial resolution down-turn).
    """
    def objective(e):
        return float(f_curve(e)) - y_obs
    if objective(lo) > 0.0:            # y_obs below the curve at lo -> clip to floor
        return lo
    h = hi
    while objective(h) < 0.0 and h < hi_max:
        h *= 2.0
    if objective(h) < 0.0:             # cannot bracket within hi_max -> unreliable
        return float("nan")
    try:
        return brentq(objective, lo, h)
    except ValueError:
        return float("nan")


def reco_metrics_over_grid(f_low, f_med, f_high,
                           e_test: np.ndarray = None):
    """Run Neyman reconstruction over a grid of true energies.

    Returns (e_test, response_ratio, resolution).

    `response_ratio` = E_reco/E_true where E_reco inverts the MEDIAN surrogate
    through ITSELF (y_obs = f_med(E_true)). This is identically 1 for any model,
    so it is a *self-consistency check of the inversion*, NOT a closure test of
    the surrogate -- treat it accordingly. `resolution` uses the Neyman crossover
    (upper E bound from inverting the lower quantile, lower bound from the upper)
    and is NaN wherever the band edge cannot be reliably inverted (saturation), so
    the reported curve is automatically restricted to the trustworthy range.
    """
    if e_test is None:
        e_test = np.linspace(10, 400, 100)
    response, resolution = [], []
    for e_true in e_test:
        y_obs = f_med(e_true)
        e_reco = invert_brent(y_obs, f_med)
        e_reco_hi = invert_brent(y_obs, f_low)   # Neyman crossover
        e_reco_lo = invert_brent(y_obs, f_high)  # Neyman crossover
        response.append(e_reco / e_true)
        resolution.append((e_reco_hi - e_reco_lo) / (2.0 * e_true))
    return e_test, np.array(response), np.array(resolution)


# ---- stochastic-term fit -----------------------------------------------------

def fit_stochastic(e_test, resolution):
    """Least-squares fit sigma/E = sqrt((a/sqrt(E))^2 + b^2 + (c/E)^2).

    `a` = stochastic term, `b` = constant term, `c` = noise term. Fits only the
    finite, positive points (so the NaN-restricted saturation tail is excluded).
    Returns dict(a, b, c) or None if too few usable points.
    """
    e = np.asarray(e_test, float)
    r = np.asarray(resolution, float)
    m = np.isfinite(e) & np.isfinite(r) & (e > 0) & (r > 0)
    if m.sum() < 4:
        return None
    def model(E, a, b, c):
        return np.sqrt((a / np.sqrt(E)) ** 2 + b ** 2 + (c / E) ** 2)
    try:
        p, _ = curve_fit(model, e[m], r[m], p0=[0.2, 0.05, 0.5],
                         bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        return {"a": float(p[0]), "b": float(p[1]), "c": float(p[2])}
    except Exception:
        return None


# ---- the 3-panel dashboard plot ----------------------------------------------

READOUT_COLORS = {
    "Analog":  "royalblue",
    "MIP":     "forestgreen",
    "Hits":    "crimson",
    "Cluster": "darkorchid",
    "Cluster (baseline)": "darkorchid",
    "Cluster (improved)": "teal",
}

READOUT_LABELS = {
    "Analog":  "True Analog",
    "MIP":     "MIP counting",
    "Hits":    "Raw Hits",
    "Cluster": "Naive 2D Clustering",
    "Cluster (baseline)": "Cluster (baseline)",
    "Cluster (improved)": "Cluster (improved)",
}


def plot_dashboard(reco_results, out_path_prefix=None, show=False):
    """Produce the 3-panel reconstruction dashboard.

    `reco_results` is a dict keyed by readout name mapping to
    (e_test, response, resolution) tuples.

    If `out_path_prefix` is given, saves to {prefix}_linearity.png and
    {prefix}_resolution.png. If `show` is True, also calls plt.show().
    """
    import matplotlib
    if out_path_prefix is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Panel 1: median self-inversion self-consistency check (NOT a closure test).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, (et, resp, _) in reco_results.items():
        ax.plot(et, resp, color=READOUT_COLORS.get(key, "gray"), lw=2,
                label=READOUT_LABELS.get(key, key))
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylim(0.8, 1.2)
    ax.set_title("Median Self-Inversion Check ($E_{reco}/E_{true}$)", fontsize=13)
    ax.set_xlabel("True Beam Energy ($E_{true}$) [GeV]")
    ax.set_ylabel("Response Ratio")
    ax.text(0.5, 0.04, "self-consistency of the inversion ($\\equiv$1 by construction); "
            "not a surrogate closure test", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, color="0.4")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path_prefix:
        plt.savefig(f"{out_path_prefix}_linearity.png", dpi=110)
    if show:
        plt.show()
    else:
        plt.close()

    # Panels 2 + 3: Resolution + Stochastic (with fitted a/sqrt(E) (+) b (+) c/E)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for key, (et, _, res) in reco_results.items():
        axes[0].plot(et, res, color=READOUT_COLORS.get(key, "gray"), lw=2,
                     label=READOUT_LABELS.get(key, key))
    axes[0].set_title("Reconstructed Resolution ($\\sigma_{reco}/E_{true}$)", fontsize=13)
    axes[0].set_xlabel("True Beam Energy ($E_{true}$) [GeV]")
    axes[0].set_ylabel("Energy Resolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key, (et, _, res) in reco_results.items():
        col = READOUT_COLORS.get(key, "gray")
        fit = fit_stochastic(et, res)
        lbl = READOUT_LABELS.get(key, key)
        if fit:
            lbl += f"  (a={fit['a']:.2f}, b={fit['b']:.3f})"
        axes[1].plot(1.0 / np.sqrt(et), res, color=col, lw=2, label=lbl)
        if fit:
            ef = np.asarray(et, float)
            ef = ef[np.isfinite(ef) & (ef > 0)]
            if ef.size:
                eg = np.linspace(ef.min(), ef.max(), 200)
                rg = np.sqrt((fit["a"] / np.sqrt(eg)) ** 2 + fit["b"] ** 2 + (fit["c"] / eg) ** 2)
                axes[1].plot(1.0 / np.sqrt(eg), rg, color=col, lw=1, ls="--", alpha=0.6)
    axes[1].set_title("Stochastic Resolution ($\\sigma/E$ vs $1/\\sqrt{E}$, fit overlaid)",
                      fontsize=13, pad=40)
    axes[1].set_xlabel("$1/\\sqrt{E_{true}}$ [GeV$^{-1/2}$]")
    axes[1].set_ylabel("Energy Resolution")

    def _to_e(x):
        return 1.0 / np.maximum(x, 1e-10) ** 2
    def _from_e(e):
        return 1.0 / np.sqrt(np.maximum(e, 1e-10))

    sec = axes[1].secondary_xaxis("top", functions=(_to_e, _from_e))
    sec.set_xlabel("True Beam Energy [GeV]")
    sec.set_xticks([400, 100, 50, 25, 10])
    sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path_prefix:
        plt.savefig(f"{out_path_prefix}_resolution.png", dpi=110)
    if show:
        plt.show()
    else:
        plt.close()


# ---- held-out event-based closure (the honest test) --------------------------

def reco_closure_events(f_med, y_obs, e_true, e_bins=None, verbose=True, label=""):
    """Honest closure on a HELD-OUT set: invert each event's OBSERVED readout through the
    median surrogate to recover E_reco, then bin by E_true. Returns (centers, response,
    resolution) with response = mean(E_reco/E_true) and resolution = std(E_reco/E_true) per
    bin -- a real test on events no ensemble member trained on, unlike the median self-inversion
    (which is 1 by construction). Uses the same invert_brent as the dashboard.

    Treat the EDGE bins with care: at the top, events whose readout fluctuated above the
    saturated curve invert to NaN and are dropped (informative censoring -- the surviving
    events understate the spread there); at the bottom, events whose readout falls below
    the curve's value at the 5 GeV bracket floor are clipped to E_reco = 5 GeV, pulling
    that bin's ratio toward 5/E_true. verbose=True prints the censored/clipped counts."""
    y_obs = np.asarray(y_obs, dtype=float); e_true = np.asarray(e_true, dtype=float)
    if e_bins is None:
        e_bins = np.linspace(float(np.nanmin(e_true)), float(np.nanmax(e_true)), 16)
    e_reco = np.array([invert_brent(float(y), f_med) for y in y_obs])
    n_nan = int(np.sum(~np.isfinite(e_reco)))
    n_floor = int(np.sum(e_reco == 5.0))       # == invert_brent's `lo` bracket floor
    if verbose and (n_nan or n_floor):
        tag = f" [{label}]" if label else ""
        print(f"  reco_closure_events{tag}: of {len(e_reco)} events, {n_nan} not invertible "
              f"(saturation -> dropped) and {n_floor} clipped at the 5 GeV floor -- "
              f"edge bins are biased accordingly")
    ratio = e_reco / e_true
    centers, resp, res = [], [], []
    for lo, hi in zip(e_bins[:-1], e_bins[1:]):
        m = (e_true >= lo) & (e_true < hi) & np.isfinite(ratio)
        if int(m.sum()) >= 5:
            centers.append(0.5 * (lo + hi))
            resp.append(float(np.mean(ratio[m])))
            res.append(float(np.std(ratio[m])))
    return np.array(centers), np.array(resp), np.array(res)


def plot_heldout_closure(heldout_results, out_path_prefix=None, show=False):
    """Two-panel honest closure from held-out events: linearity <E_reco/E_true> (left) and
    resolution std(E_reco/E_true) (right), per readout."""
    import matplotlib
    if out_path_prefix is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, (et, resp, _res) in heldout_results.items():
        if len(et):
            axes[0].plot(et, resp, "o-", color=READOUT_COLORS.get(key, "gray"), lw=2,
                         label=READOUT_LABELS.get(key, key))
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylim(0.8, 1.2)
    axes[0].set_title("Held-out linearity  $\\langle E_{reco}/E_{true}\\rangle$", fontsize=13)
    axes[0].set_xlabel("True Beam Energy [GeV]"); axes[0].set_ylabel("Response Ratio")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for key, (et, _resp, res) in heldout_results.items():
        if len(et):
            axes[1].plot(et, res, "o-", color=READOUT_COLORS.get(key, "gray"), lw=2,
                         label=READOUT_LABELS.get(key, key))
    axes[1].set_title("Held-out resolution  $\\mathrm{std}(E_{reco}/E_{true})$", fontsize=13)
    axes[1].set_xlabel("True Beam Energy [GeV]"); axes[1].set_ylabel("Energy Resolution")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path_prefix:
        plt.savefig(f"{out_path_prefix}_heldout.png", dpi=110)
    if show:
        plt.show()
    else:
        plt.close()
