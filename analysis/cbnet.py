"""Crystal-Ball Density Network surrogate for DECAL response distributions.

A `CBNet` maps normalized true energy to the four Crystal-Ball parameters
(mu, sigma, alpha, n) of the readout response, trained by **maximum likelihood**
(NLL of the observed readouts under the Crystal Ball). It is literally a
"Crystal-Ball fit done inside the network", learned as a smooth function of
energy from the whole spectrum -- so:
  * its Gaussian-core sigma is defined IDENTICALLY to the per-energy CB fits
    in 03d/03c, and
  * the same CB's power-law tail gives the tail-inclusive effective width.
Both readout-space widths are then turned into an ENERGY resolution by inverting
the network's learned calibration mu(E) with the SAME code 03c uses
(`decal_cbfit.build_calibration`), so the resolutions are directly comparable to
03c's inverted sigma_E/E and to nb03's Neyman band.

The CB is parametrized exactly as scipy.stats.crystalball (beta=alpha, m=n), the
same shape 03c fits, so a number-for-number comparison is meaningful.

Mirrors quantilenet.py: same `train_one_ensemble(x, y, ...)` signature, same y/x
normalization, same save/load bundle -- so 03d differs from nb03 only in the
model and the resolution extraction. Backbone is the same tiny MLP as QuantileNet.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

_SQRT2 = math.sqrt(2.0)
_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)

# alpha/n kept inside scipy's numerically stable domain for crystalball.ppf. The LOWER
# bounds are minimal (an aggressive floor would clip a genuine heavy leakage tail and bias
# the effective width LOW at high energy where the tail matters). The UPPER n bound is the
# important one: scipy.stats.crystalball.ppf OVERFLOWS to -inf for large n at small alpha
# (the (n/alpha)**n term), and since the central-68 half-width is essentially n-independent
# for n >~ 15 (a large n just means "light, near-Gaussian tail"), capping n at 20 is
# nearly harmless: the central-68 half-width changes by <1% for alpha >~ 1 (the normal EM
# regime) and by at most ~3% for the small-alpha (heavy, early-starting) tails the alpha=0.05
# floor permits -- and in the CONSERVATIVE (slightly wider) direction, which the ensemble
# median dilutes further. A finite-guard in resolution_over_grid is a belt-and-suspenders
# backstop (with n <= 20 scipy's ppf stays finite, so it rarely fires).
_ALPHA_CLIP = (0.05, 10.0)
_N_CLIP = (1.02, 20.0)


class CBNet(nn.Module):
    """1 input (normalized E_true) -> Crystal-Ball (mu, sigma, alpha, n)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 4),
        )
        with torch.no_grad():
            self.net[-1].bias[0] += 0.5     # start mu near the middle of the response range

    def forward(self, x: torch.Tensor):
        o = self.net(x)
        mu = o[:, 0:1]
        sigma = torch.nn.functional.softplus(o[:, 1:2]) + 1e-3
        alpha = torch.nn.functional.softplus(o[:, 2:3]) + 1e-2      # tail start, > 0
        n = torch.nn.functional.softplus(o[:, 3:4]) + 1.01          # tail power, > 1
        return mu, sigma, alpha, n


def cb_nll(params, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of `y` under the low-tail Crystal Ball (scipy convention)."""
    mu, sigma, alpha, n = params
    t = (y - mu) / sigma
    # normalization  N = 1 / (C + D)
    C = n / (alpha * (n - 1.0)) * torch.exp(-0.5 * alpha ** 2)      # tail integral (n>1)
    D = _SQRT_HALF_PI * (1.0 + torch.erf(alpha / _SQRT2))           # core integral
    logN = -torch.log(C + D)
    core = -0.5 * t ** 2
    logA = n * torch.log(n / alpha) - 0.5 * alpha ** 2
    B = n / alpha - alpha
    tail = logA - n * torch.log(torch.clamp(B - t, min=1e-9))
    logf = torch.where(t > -alpha, core, tail)
    logpdf = logN - torch.log(sigma) + logf
    return -logpdf.mean()


def save_ensemble(models, x_max, y_frac_max, filepath):
    torch.save({"models_state_dict": [m.state_dict() for m in models],
                "x_max": float(x_max), "y_frac_max": float(y_frac_max),
                "num_models": len(models)}, filepath)


def load_ensemble(filepath, device):
    ck = torch.load(filepath, weights_only=True, map_location=device)
    models = []
    for sd in ck["models_state_dict"]:
        m = CBNet().to(device); m.load_state_dict(sd); m.eval(); models.append(m)
    return models, float(ck["x_max"]), float(ck["y_frac_max"])


# ---- training (same structure/signature as quantilenet.train_one_ensemble) ----

def train_one_ensemble(x_data, y_data, device, name="ensemble",
                       num_models=20, epochs=5000, lr=0.01, patience=500,
                       seed_base=0, verbose=True):
    """Train an ensemble of CBNets modelling p(readout | E). Learns the density of
    the fractional response y/x, normalized by its max (same as quantilenet).
    Returns (models, x_max, y_frac_max)."""
    import copy, time
    import numpy as np
    import torch.optim as optim

    y_frac = y_data / x_data
    x_max, y_frac_max = float(np.max(x_data)), float(np.max(y_frac))
    x_norm = (x_data / x_max).astype(np.float32)
    y_norm = (y_frac / y_frac_max).astype(np.float32)
    if verbose:
        print(f"=== CB-net ensemble: {name}  (N={len(x_data)}, x_max={x_max:.3g}, y_frac_max={y_frac_max:.3g}) ===")

    models = []
    t_total = time.time()
    for m_idx in range(num_models):
        rng = np.random.RandomState(seed_base + m_idx)
        perm = rng.permutation(len(x_norm)); split = int(len(x_norm) * 0.8)
        tr, va = perm[:split], perm[split:]
        Xt = torch.tensor(x_norm[tr], device=device).unsqueeze(1)
        Yt = torch.tensor(y_norm[tr], device=device).unsqueeze(1)
        Xv = torch.tensor(x_norm[va], device=device).unsqueeze(1)
        Yv = torch.tensor(y_norm[va], device=device).unsqueeze(1)

        m = CBNet().to(device)
        opt = optim.Adam(m.parameters(), lr=lr)
        best_val, best_w, no_improve, last = float("inf"), None, 0, 0
        t0 = time.time()
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            loss = cb_nll(m(Xt), Yt)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            opt.step()
            m.eval()
            with torch.no_grad():
                vl = cb_nll(m(Xv), Yv).item()
            if vl < best_val - 1e-6:
                best_val, best_w, no_improve = vl, copy.deepcopy(m.state_dict()), 0
            else:
                no_improve += 1
            last = ep
            if no_improve >= patience:
                break
        if best_w is not None:
            m.load_state_dict(best_w)
        else:
            # never found a finite validation loss (diverged on epoch 0): this member is
            # untrained random weights -- flag it so the caller/median can see it happened.
            if verbose:
                print(f"  model {m_idx+1:2d}/{num_models}  WARNING: no finite val loss; member left untrained")
        m.eval(); models.append(m)
        if verbose:
            print(f"  model {m_idx+1:2d}/{num_models}  best_nll={best_val:.4f}  epochs={last+1:5d}  ({time.time()-t0:.1f}s)")
    if verbose:
        print(f"  ensemble total: {time.time()-t_total:.1f}s")
    return models, x_max, y_frac_max


# ---- CB parameter curves + resolution extraction -----------------------------

def _member_params(ensemble, x_max, y_frac_max, device, e_grid):
    """Per-member CB params over the energy grid, in FRACTIONAL-RESPONSE (y/E) units.
    Returns lists (mu, sigma, alpha, n), each entry an array over e_grid. alpha,n are
    clipped to the scipy-stable domain (_ALPHA_CLIP, _N_CLIP)."""
    import numpy as np
    xin = torch.tensor((np.asarray(e_grid, float) / x_max)[:, None], dtype=torch.float32, device=device)
    mus, sigs, als, ns = [], [], [], []
    for m in ensemble:
        with torch.no_grad():
            mu, sigma, alpha, n = m(xin)
        mus.append(mu.cpu().numpy().ravel() * y_frac_max)
        sigs.append(sigma.cpu().numpy().ravel() * y_frac_max)
        als.append(np.clip(alpha.cpu().numpy().ravel(), *_ALPHA_CLIP))
        ns.append(np.clip(n.cpu().numpy().ravel(), *_N_CLIP))
    return mus, sigs, als, ns


def cb_params_over_grid(ensemble, x_max, y_frac_max, device, e_grid):
    """Ensemble-typical CB params vs energy, in FRACTIONAL-RESPONSE (y/E) units, for the
    density-overlay plots. mu,sigma are the ensemble MEAN; the shape params alpha,n use the
    ensemble MEDIAN (robust to a member with a runaway tail), both after the same [_ALPHA_CLIP,
    _N_CLIP] clip resolution_over_grid uses -- so the plotted CB (03d section 5) and the
    effective width (section 7) correspond to a consistent shape."""
    import numpy as np
    mus, sigs, als, ns = _member_params(ensemble, x_max, y_frac_max, device, e_grid)
    return {"E": np.asarray(e_grid, float),
            "mu": np.mean(mus, 0), "sigma": np.mean(sigs, 0),
            "alpha": np.median(als, 0), "n": np.median(ns, 0)}


def resolution_over_grid(ensemble, x_max, y_frac_max, device, e_grid):
    """Energy-domain resolutions sigma_E/E vs E from the CB-density ensemble.

    Returns (cb, core, eff) where core and eff are FRACTIONAL ENERGY resolutions
    (sigma_E/E), NOT raw readout widths. Both are obtained by INVERTING the network's
    learned calibration mu(E) exactly as 03c inverts its measured calibration
    (decal_cbfit.build_calibration), so they are directly comparable to 03c's inverted
    sigma_E/E and to nb03's Neyman band:

      core = (ginv(mu_r + sc_r) - ginv(mu_r - sc_r)) / (2E)
             Gaussian-core energy resolution  (excludes the low-side tail; == 03c core)
      eff  = (ginv(readout_P84) - ginv(readout_P16)) / (2E)
             tail-inclusive energy resolution (includes the leakage tail; ~ nb03 band)

    For the linear analog readout the inversion is ~identity (sigma_E/E ~ sigma/mu); for the
    SATURATING digital readouts it is essential -- a small readout width maps to a large energy
    interval where the calibration has flattened, so sigma/mu would UNDER-report the resolution.

    Widths are the ALEATORIC response width: computed per ensemble member (in readout units),
    combined by the ensemble nanMEDIAN, and only THEN inverted through the ensemble-mean
    calibration. The nanmedian is the PRIMARY safeguard against a pathological member (robust to
    a minority of heavy-tail members, and -- unlike pooling the members' samples -- not inflated
    by epistemic disagreement). Two secondary guards only keep a single member's width FINITE
    before the median sees it: the n<=20 clip stops scipy's ppf overflowing, and a PHYSICAL floor
    holds the P16 readout above ~0 (it floors only the LOW edge, not P84); the finite-guard is
    then belt-and-suspenders that rarely fires. Note the floor bounds the inversion away from
    -inf but does NOT by itself make a heavy-tail member's width sensible -- the median across
    members is what does that, so do not remove it. Also returns the ensemble-typical CB params
    for the density/calibration plots."""
    import numpy as np
    from scipy.stats import crystalball
    from scipy.ndimage import median_filter
    from decal_cbfit import build_calibration

    E = np.asarray(e_grid, float)
    mus, sigs, als, ns = _member_params(ensemble, x_max, y_frac_max, device, e_grid)

    # per-member readout-space band (fractional-response units * E = readout units)
    mu_r_m, sc_r_m, p16_m, p84_m = [], [], [], []
    for mu, sigma, al, nn in zip(mus, sigs, als, ns):
        mu_r = mu * E                                   # readout-units mean (calibration point)
        scale = sigma * E                               # CB scale in readout units
        off_lo = crystalball.ppf(0.1587, al, nn) * scale   # readout offset of P16 (<0)
        off_hi = crystalball.ppf(0.8413, al, nn) * scale   # readout offset of P84 (>0)
        # finite-guard: any residual scipy ppf overflow (-inf) is dropped, not floored, so it
        # cannot bias the median -- nanmedian simply ignores that member at that energy.
        off_lo = np.where(np.isfinite(off_lo), off_lo, np.nan)
        off_hi = np.where(np.isfinite(off_hi), off_hi, np.nan)
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(mu_r) | (mu_r <= 0)          # mu is unconstrained -> guard mu<=0
            mu_r = np.where(bad, np.nan, mu_r)
            p16 = np.maximum(mu_r + off_lo, 0.02 * mu_r)    # PHYSICAL floor, LOW edge only: readout >~ 0
            p84 = mu_r + off_hi                             # P84 unfloored (the median tames outliers)
        mu_r_m.append(mu_r); sc_r_m.append(sigma * E); p16_m.append(p16); p84_m.append(p84)

    with np.errstate(invalid="ignore"):
        mu_r = np.nanmean(mu_r_m, axis=0)               # ensemble-mean calibration (consensus)
        sc_r = np.nanmedian(sc_r_m, axis=0)             # aleatoric core width (readout units)
        p16 = np.nanmedian(p16_m, axis=0)
        p84 = np.nanmedian(p84_m, axis=0)

    core = np.full_like(E, np.nan, dtype=float)
    eff = np.full_like(E, np.nan, dtype=float)
    good = np.isfinite(mu_r) & (mu_r > 0)
    if good.sum() >= 2:
        _, ginv, _, _ = build_calibration(E[good], mu_r[good])
        Eg = E[good]; mr = mu_r[good]
        lo_core = np.maximum(mr - sc_r[good], 1e-9)
        core[good] = (ginv(mr + sc_r[good]) - ginv(lo_core)) / (2.0 * Eg)
        eff[good] = (ginv(np.maximum(p84[good], 1e-9)) - ginv(np.maximum(p16[good], 1e-9))) / (2.0 * Eg)

    # light smoothing of residual single-point ppf/inversion outliers (small filter; the
    # caller's grid is log-spaced so a size-3 window is a fixed, narrow fraction of a decade)
    if np.isfinite(eff).sum() >= 5:
        eff = median_filter(np.nan_to_num(eff, nan=float(np.nanmedian(eff))), size=3, mode="nearest")
        core = median_filter(np.nan_to_num(core, nan=float(np.nanmedian(core))), size=3, mode="nearest")

    cb = {"E": E, "mu": np.nanmean(mus, 0), "sigma": np.nanmean(sigs, 0),
          "alpha": np.nanmedian(als, 0), "n": np.nanmedian(ns, 0)}
    return cb, core, eff
