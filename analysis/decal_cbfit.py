"""Shared conventional Crystal-Ball fit + calibration inversion for the DECAL
resolution notebooks.

Both the conventional notebook (03c) and the CB-density-net notebook (03d) quote a
resolution the same two-step way:
  1. fit a low-tail Crystal Ball to a response distribution -> core width sigma, and
  2. invert the calibration mu(E) to turn that READOUT width into an ENERGY
     resolution sigma_E/E (essential for the saturating digital readouts, where a
     small readout width maps to a large energy interval).

Keeping the estimator here -- imported by 03c, 03d, and analysis/cbnet.py -- guarantees
the "conventional (03c)" numbers 03d overlays are produced by the SAME code as 03c, not a
re-implementation that can silently drift.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import crystalball, norm
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator


def fit_response(values, nbins=40, min_events=300):
    """Binned single-sided (low-tail) Crystal-Ball fit of a response distribution.
    Returns dict(mu, sigma, sigma_err, beta, m, model, ok, n). Never raises. The
    graceful-degrade ladder is Crystal Ball -> Gaussian -> mean/RMS, and it triggers on
    DISCRETENESS too: small-integer count distributions (hit/cluster counts at very low
    energy) are Poisson-like, not a low-side-tail shape, so a CB is the wrong model and we
    quote the RMS instead. sigma_err is the fit uncertainty on the width (from the
    covariance), used to put error bars on the resolution and weight the sqrt(E) fit."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    n = v.size
    out = {"n": n, "ok": False, "model": "none", "mu": np.nan, "sigma": np.nan,
           "sigma_err": np.nan, "beta": np.nan, "m": np.nan}
    def rms():
        m_, s_ = float(np.mean(v)), float(np.std(v))
        out.update(mu=m_, sigma=s_, sigma_err=s_ / np.sqrt(2 * max(n, 1)), model="rms", ok=s_ > 0)
        return out
    if n < min_events or np.ptp(v) == 0:
        return rms() if n else out
    # discreteness guard: too few distinct values (or small-integer counts) -> RMS, not CB
    nuniq = np.unique(v).size
    if nuniq < 12 or (np.allclose(v, np.round(v)) and nuniq < 25):
        return rms()
    lo, hi = np.percentile(v, [0.3, 99.7])
    if hi <= lo:
        return rms()
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:]); bw = edges[1] - edges[0]
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826
    sig0 = min(max(mad if mad > 0 else float(np.std(v)), bw), hi - lo)   # keep p0 inside bounds
    N0 = n * bw; yerr = np.sqrt(counts + 1.0)
    def cb(x, N, beta, mm, loc, scale):
        with np.errstate(over="ignore", invalid="ignore"):
            return N * crystalball.pdf(x, beta, mm, loc=loc, scale=scale)
    try:
        p, pcov = curve_fit(cb, centers, counts, p0=[N0, 1.5, 5.0, med, sig0], sigma=yerr,
                            absolute_sigma=True,
                            bounds=([0, 0.1, 1.001, lo, 1e-9], [np.inf, 25, 20, hi, hi - lo]),
                            maxfev=30000)
        N, beta, mm, loc, scale = p
        serr = float(np.sqrt(pcov[4, 4])) if np.all(np.isfinite(pcov)) else scale / np.sqrt(2 * n)
        # reject a runaway fit (core wider than the mean is not a sensible resolution -- happens
        # for a bimodal / non-single-peaked response, e.g. a hadron punching through)
        if 0 < scale < min(hi - lo, loc) and beta > 0.11 and lo < loc < hi:
            out.update(mu=float(loc), sigma=float(scale), sigma_err=serr, beta=float(beta),
                       m=float(mm), model="crystalball", ok=True)
            return out
    except Exception:
        pass
    def gau(x, N, loc, scale):
        return N * norm.pdf(x, loc=loc, scale=scale)
    try:
        p, pcov = curve_fit(gau, centers, counts, p0=[N0, med, sig0], sigma=yerr,
                            absolute_sigma=True, bounds=([0, lo, 1e-9], [np.inf, hi, hi - lo]),
                            maxfev=30000)
        serr = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else p[2] / np.sqrt(2 * n)
        if 0 < p[2] < p[1]:                              # same sanity gate: sigma < mu
            out.update(mu=float(p[1]), sigma=float(p[2]), sigma_err=serr, model="gaussian", ok=True)
            return out
    except Exception:
        pass
    return rms()


def build_calibration(E, mu, pad=1.0, n=2000):
    """Monotonic calibration mu(E) (PCHIP in log-log) and its inverse ginv(readout)->E.
    ginv extrapolates LINEARLY in log-log beyond the fitted range instead of clamping to the
    grid edge -- so an up-fluctuation mu+sigma above mu(E_max) still inverts to an energy
    above E_max rather than being silently pinned to E_max (which would HIDE the saturation
    the whole notebook is about). Strict monotonicity guarantees a single-valued inverse."""
    E = np.asarray(E, float); mu = np.asarray(mu, float)
    o = np.argsort(E); lE, lmu = np.log(E[o]), np.log(mu[o])
    if np.unique(lE).size < 2:
        raise ValueError("need >= 2 distinct energy points to build a calibration")
    fwd = PchipInterpolator(lE, lmu, extrapolate=True)
    lEg = np.linspace(lE.min() - pad, lE.max() + pad, n)
    lmug = np.maximum.accumulate(fwd(lEg)) + 1e-9 * np.arange(n)   # STRICTLY increasing
    sL = (lEg[1] - lEg[0]) / (lmug[1] - lmug[0])          # boundary slopes for extrapolation
    sR = (lEg[-1] - lEg[-2]) / (lmug[-1] - lmug[-2])
    def ginv(o):
        z = np.log(np.clip(np.asarray(o, float), 1e-30, None))
        r = np.interp(z, lmug, lEg)
        r = np.where(z < lmug[0],  lEg[0]  + (z - lmug[0])  * sL, r)
        r = np.where(z > lmug[-1], lEg[-1] + (z - lmug[-1]) * sR, r)
        return np.exp(r)
    g = lambda E: np.exp(fwd(np.log(np.clip(np.asarray(E, float), 1e-30, None))))
    return g, ginv, np.exp(lEg), np.exp(lmug)
