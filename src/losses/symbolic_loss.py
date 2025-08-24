#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbolic_loss.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)
Neuro‑symbolic composite loss with Gaussian Log‑Likelihood (GLL) + physics‑informed penalties.

Design goals
------------
• Primary probabilistic objective: Gaussian Log‑Likelihood on (μ, σ) for each of 283 spectral bins
  (competition uses a GLL‑style score; over‑confidence is penalized) [oai_citation:0‡Using Hugging Face for the NeurIPS Ariel Data Challenge 2025.pdf](file-service://file-3CcY5M6ouKppUqaTiLE9zR) [oai_citation:1‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).
• Symbolic/physics‑informed regularizers (toggleable via config):
  – Smoothness in wavelength (finite‑difference curvature, optional FFT high‑frequency suppression) [oai_citation:2‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).
  – Non‑negativity of spectra (penalize μ < 0) – optional, safe with soft clamp.
  – Asymmetry penalty (discourage unphysical left↔right spectral asymmetry when applicable) [oai_citation:3‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).
  – (Optional extension point) Photonic/temporal alignment hooks (e.g., transit‑phase priors).
• Full decomposition: returns total loss and per‑term components for diagnostics & dashboards.
• Differentiable end‑to‑end; numerically safe (σ clamping; eps).

References (project docs / design notes)
----------------------------------------
• Ariel challenge uncertainty metric (GLL) and need for calibrated σ [oai_citation:4‡Using Hugging Face for the NeurIPS Ariel Data Challenge 2025.pdf](file-service://file-3CcY5M6ouKppUqaTiLE9zR).
• CLI‑first, reproducibility, telemetry rationale (Hydra + logs) used by training pipeline [oai_citation:5‡Ubuntu CLI-Driven Architecture for Large-Scale Scientific Data Pipelines (NeurIPS 2025 Ariel Challen.pdf](file-service://file-Fdr46UbCyD9vDBpXSk9Yi1) [oai_citation:6‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).
• Pattern/smoothness rationale across spectral dimension (high‑frequency suppression) [oai_citation:7‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).

Usage
-----
    from src.losses.symbolic_loss import SymbolicLoss
    criterion = SymbolicLoss(cfg.loss)          # cfg.loss is an OmegaConf/DotDict or plain dict
    loss = criterion(mu, sigma, target)         # all tensors: [B, 283]
    # Optionally examine per-term stats for logging:
    stats = criterion.last_stats                # dict of scalars (floats) for current forward()

Notes
-----
• All terms are averaged over batch unless otherwise stated.
• This file is framework-agnostic beyond PyTorch and can be unit-tested standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmoothnessConfig:
    """Configuration for spectral smoothness regularization."""
    enable_fd2: bool = True              # 2nd-derivative finite-difference penalty
    fd2_weight: float = 1.0e-3
    enable_fft: bool = False             # High-frequency FFT power penalty
    fft_weight: float = 0.0
    fft_cutoff_ratio: float = 0.5        # keep frequencies > cutoff as "high-frequency" (0..1)
    fft_power: float = 2.0               # Lp power on magnitudes in the HF band (2.0 = L2)


@dataclass
class AsymmetryConfig:
    """Configuration for asymmetry penalty (μ vs. reversed μ across wavelength)."""
    enable: bool = False
    weight: float = 0.0
    p: float = 2.0                        # Lp on |μ - reverse(μ)| (2.0 = L2)


@dataclass
class NonNegConfig:
    """Configuration for (soft) non-negativity penalty on μ."""
    enable: bool = False
    weight: float = 0.0
    p: float = 2.0                        # Lp on relu(-μ)


@dataclass
class GLLConfig:
    """Configuration for Gaussian Log-Likelihood term."""
    weight: float = 1.0
    sigma_min: float = 1.0e-6            # numerical floor for σ
    sigma_scale: float = 1.0             # global temperature scaling on σ (posthoc calibration) [oai_citation:8‡AI Design and Modeling.pdf](file-service://file-6oS7N1e7T9DKuWz68BoAPi)


@dataclass
class SymbolicLossConfig:
    """Top-level configuration for SymbolicLoss."""
    # GLL (primary objective)
    gll: GLLConfig = GLLConfig()
    # Smoothness families
    smooth: SmoothnessConfig = SmoothnessConfig()
    # Asymmetry (optional)
    asym: AsymmetryConfig = AsymmetryConfig()
    # Non-negativity (optional)
    nonneg: NonNegConfig = NonNegConfig()
    # Reduction across batch: 'mean' or 'sum'
    reduction: str = "mean"


def _to_cfg_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf/argparse Namespace/plain-dict into plain dict of dicts."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # OmegaConf-like object: try attribute access
    out: Dict[str, Any] = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:  # pragma: no cover
            continue
        if callable(v):
            continue
        out[k] = v
    return out


class SymbolicLoss(nn.Module):
    """
    Composite neuro‑symbolic loss: GLL(μ, σ; y) + Σ λ_i * regularizer_i(μ).

    Forward:
        loss = w_gll * L_gll + w_fd2 * L_fd2 + w_fft * L_fft + w_nonneg * L_nonneg + w_asym * L_asym
        Returns a scalar tensor with gradients. Per-term scalars in self.last_stats for logging.

    Expected shapes:
        mu:    [B, D]    (D=283 bins)
        sigma: [B, D]    (strictly positive; internally clamped)
        y:     [B, D]
    """

    def __init__(self, cfg: Optional[Any] = None):
        super().__init__()

        # Parse/normalize config into dataclasses (supports dict / OmegaConf / None)
        cfgd = _to_cfg_dict(cfg)
        self.cfg = SymbolicLossConfig(
            gll=GLLConfig(**cfgd.get("gll", {})),
            smooth=SmoothnessConfig(**cfgd.get("smooth", {})),
            asym=AsymmetryConfig(**cfgd.get("asym", {})),
            nonneg=NonNegConfig(**cfgd.get("nonneg", {})),
            reduction=cfgd.get("reduction", "mean"),
        )

        if self.cfg.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")

        # Last forward-pass stats for logging/diagnostics
        self.last_stats: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute composite loss. All tensors should be on the same device/dtype.
        """
        self._validate_inputs(mu, sigma, target)

        # Primary probabilistic loss: Gaussian negative log-likelihood (per bin) [oai_citation:9‡Using Hugging Face for the NeurIPS Ariel Data Challenge 2025.pdf](file-service://file-3CcY5M6ouKppUqaTiLE9zR)
        l_gll = self._gll(mu, sigma, target)

        # Spectral smoothness penalties
        l_fd2 = self._fd2_penalty(mu) if self.cfg.smooth.enable_fd2 else mu.new_tensor(0.0)
        l_fft = self._fft_penalty(mu) if self.cfg.smooth.enable_fft else mu.new_tensor(0.0)

        # Optional constraints
        l_nonneg = self._nonneg_penalty(mu) if self.cfg.nonneg.enable else mu.new_tensor(0.0)
        l_asym = self._asym_penalty(mu) if self.cfg.asym.enable else mu.new_tensor(0.0)

        # Weighted sum
        loss = (
            self.cfg.gll.weight * l_gll
            + self.cfg.smooth.fd2_weight * l_fd2
            + self.cfg.smooth.fft_weight * l_fft
            + self.cfg.nonneg.weight * l_nonneg
            + self.cfg.asym.weight * l_asym
        )

        # Save scalar stats for external logging (convert to floats safely)
        to_float = lambda t: float(t.detach().item())
        self.last_stats = {
            "loss_total": to_float(loss),
            "loss_gll": to_float(l_gll),
            "loss_fd2": to_float(l_fd2),
            "loss_fft": to_float(l_fft),
            "loss_nonneg": to_float(l_nonneg),
            "loss_asym": to_float(l_asym),
            "gll_sigma_scale": float(self.cfg.gll.sigma_scale),
            "gll_sigma_min": float(self.cfg.gll.sigma_min),
        }

        return loss

    # -------------------------------------------------------------------------
    # Core terms
    # -------------------------------------------------------------------------
    def _gll(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gaussian negative log-likelihood on (μ, σ) for each bin, averaged per cfg.reduction.

        NLL per element: 0.5 * [ log(2πσ^2) + ((y - μ)^2) / σ^2 ]
        """
        # Post-hoc temperature scaling & numerical floor on σ [oai_citation:10‡AI Design and Modeling.pdf](file-service://file-6oS7N1e7T9DKuWz68BoAPi)
        s = torch.clamp(sigma * self.cfg.gll.sigma_scale, min=self.cfg.gll.sigma_min)
        var = s * s
        nll = 0.5 * (torch.log(2.0 * torch.pi * var) + (y - mu) ** 2 / var)

        if self.cfg.reduction == "mean":
            return nll.mean()
        else:
            return nll.sum()

    def _fd2_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Finite-difference curvature (second derivative) penalty across wavelength bins.
        Encourages smooth spectra (physics prior: neighboring wavelengths are correlated) [oai_citation:11‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).
        """
        # mu: [B, D]
        # Second difference: μ_{i+1} - 2μ_i + μ_{i-1} for interior points
        mu_left = mu[..., :-2]
        mu_mid = mu[..., 1:-1]
        mu_right = mu[..., 2:]
        curv = mu_right - 2.0 * mu_mid + mu_left               # [B, D-2]
        val = (curv ** 2).mean() if self.cfg.reduction == "mean" else (curv ** 2).sum()
        return val

    def _fft_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Penalize high-frequency power in spectra via rFFT magnitude above cutoff ratio.
        Uses Lp power on magnitudes in the HF band (default L2) [oai_citation:12‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).

        Notes:
            • Treats last dimension as wavelength; batch is aggregated per reduction.
            • fft_cutoff_ratio ∈ [0,1]. If 0.5, the upper half of frequencies is penalized.
        """
        # Real FFT along wavelength dimension
        # rfft output length N_fft = floor(D/2) + 1
        x = mu - mu.mean(dim=-1, keepdim=True)  # remove DC bias before FFT to focus on shape
        hf = torch.fft.rfft(x, dim=-1)          # [B, N_fft], complex

        B, N_fft = hf.shape
        if N_fft <= 1:
            return mu.new_tensor(0.0)

        # Build boolean high-frequency mask
        cutoff_idx = int(self.cfg.smooth.fft_cutoff_ratio * (N_fft - 1))
        cutoff_idx = max(0, min(N_fft - 1, cutoff_idx))
        # Penalize strictly above cutoff
        mask = torch.zeros(N_fft, dtype=torch.bool, device=mu.device)
        if cutoff_idx + 1 < N_fft:
            mask[cutoff_idx + 1 :] = True

        mag = torch.abs(hf)  # [B, N_fft]
        if self.cfg.smooth.fft_power == 2.0:
            powv = (mag[..., mask] ** 2).sum(dim=-1)  # [B]
        else:
            powv = (mag[..., mask].abs() ** self.cfg.smooth.fft_power).sum(dim=-1)  # [B]

        val = powv.mean() if self.cfg.reduction == "mean" else powv.sum()
        return val

    def _nonneg_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Penalize negative spectral values softly: Lp on relu(-μ).
        """
        neg = F.relu(-mu)  # only where μ < 0 contributes
        if self.cfg.nonneg.p == 1.0:
            v = neg.abs().sum(dim=-1)
        elif self.cfg.nonneg.p == 2.0:
            v = (neg ** 2).sum(dim=-1)
        else:
            v = (neg.abs() ** self.cfg.nonneg.p).sum(dim=-1)

        return v.mean() if self.cfg.reduction == "mean" else v.sum()

    def _asym_penalty(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Penalize left↔right asymmetry by comparing μ to its reversed copy across wavelength bins.
        Useful when spectra are expected to be approximately symmetric around central features (optional) [oai_citation:13‡Patterns, Algorithms, and Fractals: A Cross-Disciplinary Technical Reference.pdf](file-service://file-J9TRNKUxjEL2k8txdJgXqf).
        """
        rev = torch.flip(mu, dims=[-1])
        diff = torch.abs(mu - rev)
        if self.cfg.asym.p == 1.0:
            v = diff.sum(dim=-1)
        elif self.cfg.asym.p == 2.0:
            v = (diff ** 2).sum(dim=-1)
        else:
            v = (diff ** self.cfg.asym.p).sum(dim=-1)

        return v.mean() if self.cfg.reduction == "mean" else v.sum()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _validate_inputs(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> None:
        if mu.shape != sigma.shape or mu.shape != target.shape:
            raise ValueError(f"mu, sigma, target must share shape [B, D]; got {mu.shape}, {sigma.shape}, {target.shape}")
        if mu.ndim != 2:
            raise ValueError(f"Expected [B, D] tensors; got mu.ndim={mu.ndim}")
        if not torch.is_floating_point(mu) or not torch.is_floating_point(sigma) or not torch.is_floating_point(target):
            raise TypeError("mu, sigma, target must be floating point tensors.")

    # -------------------------------------------------------------------------
    # Optional extension points (placeholders for future physics hooks)
    # -------------------------------------------------------------------------
    def photonic_alignment_penalty(self, *args, **kwargs) -> torch.Tensor:
        """
        Placeholder: implement alignment penalty to external photometry/transit phase if provided.
        By default returns zero; wire into forward() if/when used.
        """
        return torch.tensor(0.0, device=kwargs.get("device", None) or "cpu")


# -----------------------------------------------------------------------------
# Simple self-test (can be executed with: python -m src.losses.symbolic_loss)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 4, 283
    mu = torch.randn(B, D)
    sigma = torch.rand(B, D) * 0.1 + 0.05
    y = mu + 0.01 * torch.randn(B, D)

    cfg = {
        "gll": {"weight": 1.0, "sigma_min": 1e-6, "sigma_scale": 1.0},
        "smooth": {"enable_fd2": True, "fd2_weight": 1e-3, "enable_fft": True, "fft_weight": 1e-4, "fft_cutoff_ratio": 0.5},
        "nonneg": {"enable": True, "weight": 1e-4, "p": 2.0},
        "asym": {"enable": False, "weight": 0.0},
        "reduction": "mean",
    }

    loss_fn = SymbolicLoss(cfg)
    loss = loss_fn(mu, sigma, y)
    print("loss_total =", loss.item())
    print(loss_fn.last_stats)