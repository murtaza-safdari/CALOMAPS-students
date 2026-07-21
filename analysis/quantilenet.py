"""Deep Quantile Ensemble surrogate model for DECAL response curves.

A `QuantileNet` is a small MLP that maps normalized true energy to three
quantile predictions (15.87%, 50.00%, 84.13%) of the readout response, in
fractional-response units. An "ensemble" is a list of these networks trained
on different random train/val splits — averaging their median predictions
gives the surrogate response, and the spread across ensemble members gives
the epistemic uncertainty.

Quantile choices match the symmetric 1-sigma percentiles of a Gaussian, but
the training (Pinball loss) does not assume Gaussianity.

Architecture (~4355 parameters):
    Linear(1 -> 32) -> SiLU -> Linear(32 -> 64) -> SiLU
                    -> Linear(64 -> 32) -> SiLU -> Linear(32 -> 3)

This module is the single source of truth for the model class. Both
analysis/train_ensembles.py and analysis/verify_ensembles.py import from here.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn


# Symmetric 1-sigma quantiles of a Gaussian — the Pinball-loss targets.
QUANTILES = (0.1587, 0.5000, 0.8413)


class QuantileNet(nn.Module):
    """Tiny MLP: 1 input (normalized E_true), 3 outputs (quantiles)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def quantile_loss(preds: torch.Tensor, target: torch.Tensor,
                  quantiles=QUANTILES) -> torch.Tensor:
    """Pinball loss summed across the three predicted quantiles.

    For each quantile q, the pinball loss penalizes under-prediction by q and
    over-prediction by (1-q), giving an asymmetric loss that converges to the
    desired percentile of the target distribution at optimum.
    """
    loss = 0.0
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=preds.device)
    for i, q in enumerate(quantiles):
        error = target - preds[:, i:i+1]
        loss = loss + torch.max(q_tensor[i] * error, (q_tensor[i] - 1.0) * error).mean()
    return loss


def save_ensemble(models: list, x_max: float, y_frac_max: float, filepath: str) -> None:
    """Bundle a list of trained models + normalization constants into one .pth file."""
    checkpoint = {
        "models_state_dict": [m.state_dict() for m in models],
        "x_max": float(x_max),
        "y_frac_max": float(y_frac_max),
        "num_models": len(models),
    }
    torch.save(checkpoint, filepath)


def load_ensemble(filepath: str, device: torch.device):
    """Rebuild an ensemble of QuantileNets from a saved checkpoint.

    Uses map_location so checkpoints serialized on GPU can be loaded on CPU.
    Returns (models_list, x_max, y_frac_max).
    """
    ck = torch.load(filepath, weights_only=True, map_location=device)
    models = []
    for state_dict in ck["models_state_dict"]:
        m = QuantileNet().to(device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
    return models, float(ck["x_max"]), float(ck["y_frac_max"])


# ---- training -----------------------------------------------------------------

def train_one_ensemble(x_data, y_data, device, name="ensemble",
                       num_models=20, epochs=5000, lr=0.01, patience=500,
                       seed_base=0, verbose=True):
    """Train an ensemble of `num_models` QuantileNets to map x_data -> y_data.

    The network actually predicts the *fractional response* y / x, normalized
    by its max value. Returns (models_list, x_max, y_frac_max) — the same
    triple `load_ensemble` returns, so you can pass it straight to the
    dashboard helpers.

    Each model gets a fresh random 80/20 train/val split (drawn without
    replacement — a subsample, not a bootstrap resample).
    Early stopping when val loss hasn't improved for `patience` epochs.
    """
    import copy, time, sys
    import numpy as np
    import torch.optim as optim

    y_frac = y_data / x_data
    x_max, y_frac_max = float(np.max(x_data)), float(np.max(y_frac))
    x_norm = (x_data / x_max).astype(np.float32)
    y_norm = (y_frac / y_frac_max).astype(np.float32)

    if verbose:
        print(f"=== ensemble: {name}  (N={len(x_data)}, x_max={x_max:.3g}, y_frac_max={y_frac_max:.3g}) ===")

    models = []
    t_total = time.time()
    for m_idx in range(num_models):
        rng = np.random.RandomState(seed_base + m_idx)
        perm = rng.permutation(len(x_norm))
        split = int(len(x_norm) * 0.8)
        tr_idx, va_idx = perm[:split], perm[split:]

        Xt = torch.tensor(x_norm[tr_idx], dtype=torch.float32, device=device).unsqueeze(1)
        Yt = torch.tensor(y_norm[tr_idx], dtype=torch.float32, device=device).unsqueeze(1)
        Xv = torch.tensor(x_norm[va_idx], dtype=torch.float32, device=device).unsqueeze(1)
        Yv = torch.tensor(y_norm[va_idx], dtype=torch.float32, device=device).unsqueeze(1)

        m = QuantileNet().to(device)
        opt = optim.Adam(m.parameters(), lr=lr)

        best_val = float("inf")
        best_w = None
        no_improve = 0
        t0 = time.time()
        last_epoch = 0

        for ep in range(epochs):
            m.train()
            opt.zero_grad()
            loss = quantile_loss(m(Xt), Yt, QUANTILES)
            loss.backward()
            opt.step()

            m.eval()
            with torch.no_grad():
                vl = quantile_loss(m(Xv), Yv, QUANTILES).item()
            if vl < best_val:
                best_val = vl
                best_w = copy.deepcopy(m.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            last_epoch = ep
            if no_improve >= patience:
                break

        m.load_state_dict(best_w)
        m.eval()
        models.append(m)
        if verbose:
            dt = time.time() - t0
            print(f"  model {m_idx+1:2d}/{num_models}  best_val={best_val:.5f}  epochs={last_epoch+1:5d}  ({dt:.1f}s)")

    if verbose:
        print(f"  ensemble total: {time.time()-t_total:.1f}s")
    return models, x_max, y_frac_max
