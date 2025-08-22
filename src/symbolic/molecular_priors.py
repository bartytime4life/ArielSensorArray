#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpectraMind V50 — Molecular Priors (symbolic module)

Purpose
-------
Encode physics-informed constraints derived from known molecular absorption
fingerprints (e.g., H2O, CO2, CH4) directly into the training objective.
This module builds wavelength masks for molecular bands and computes
differentiable prior losses (band-consistency, smoothness, monotonically
approaching edges, optional Voigt/Gauss-like shape matching) to regularize
the predicted transmission spectrum μ(λ) toward physically plausible structure.

Notes
-----
• Model-agnostic: pass predicted μ (shape [B, L] or [L]), wavelength grid [L] in μm.
• All penalties are differentiable and return (loss, components_dict).
• Orientation: in transit spectroscopy, larger μ ⇒ deeper absorption (greater transit
  depth). If your convention differs, flip signs via config ('expect_higher_mu_in_band').

Integration
-----------
• Use standalone via MolecularPriors.compute_molecular_prior_loss(...)
• Or wrap with MolecularPriorLoss (see src/symbolic/losses/molecular_prior_loss.py)
• Defaults in configs/symbolic/molecular_priors.yaml

License
-------
MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class MolecularBand:
    """
    Defines a molecular absorption band by one or more wavelength windows.

    Attributes
    ----------
    name : str
        Molecule/band label (e.g., 'H2O_1p4', 'CO2_4p3').
    windows_um : list[tuple[float, float]]
        List of wavelength intervals [λ_min, λ_max] in microns that define the band support.
    weight : float
        Base band-level weight multiplier (relative importance).
    center_lines_um : list[float] | None
        Optional list of individual line centers (μm) for Voigt/Gauss-like guidance.
    half_width_um : float
        Half-width at half-maximum used for Gaussian kernel (if center_lines_um provided).
    """
    name: str
    windows_um: List[Tuple[float, float]]
    weight: float = 1.0
    center_lines_um: Optional[List[float]] = None
    half_width_um: float = 0.02  # ~20 nm by default


@dataclass
class MolecularPrioriConfig:
    """
    Configuration for molecular priors.

    Attributes
    ----------
    expect_higher_mu_in_band : bool
        If True (default), expect μ(band) >= μ(local continuum).
        If False, expect μ(band) <= μ(local continuum).
    band_consistency_weight : float
        Penalizes violation of band vs. local-continuum expectation.
    smoothness_weight : float
        L2 second-difference penalty within each band (curvature).
    edge_monotonicity_weight : float
        Encourages monotonic approach toward band interior.
    voigt_like_weight : float
        Correlation penalty nudging band shapes toward Gaussian/Voigt-like templates.
    continuum_window_um : float
        Size of the continuum ring around each band (μm) for local continuum estimate.
    min_valid_points_per_band : int
        Skip bands with too few bins on coarse grids.
    """
    expect_higher_mu_in_band: bool = True
    band_consistency_weight: float = 1.0
    smoothness_weight: float = 0.2
    edge_monotonicity_weight: float = 0.1
    voigt_like_weight: float = 0.0
    continuum_window_um: float = 0.05
    min_valid_points_per_band: int = 4


@dataclass
class MolecularPriors:
    """
    Container for bands and hyperparameters; computes molecular prior loss.
    """
    bands: List[MolecularBand] = field(default_factory=list)
    cfg: MolecularPrioriConfig = field(default_factory=MolecularPrioriConfig)

    def build_masks(self, wavelength_um: Tensor) -> Dict[str, Tensor]:
        """
        Build boolean masks per band over the provided wavelength grid.

        Parameters
        ----------
        wavelength_um : Tensor
            Shape [L], wavelengths in microns (float tensor).

        Returns
        -------
        dict[str, Tensor]
            name -> mask (bool tensor [L])
        """
        masks: Dict[str, Tensor] = {}
        for band in self.bands:
            mask = torch.zeros_like(wavelength_um, dtype=torch.bool)
            for lo, hi in band.windows_um:
                mask = mask | ((wavelength_um >= lo) & (wavelength_um <= hi))
            masks[band.name] = mask
        return masks

    def _continuum_mask(self, wavelength_um: Tensor, band_mask: Tensor, window: float) -> Tensor:
        """
        Build a continuum mask adjacent to band edges by a fixed window size.

        Selects wavelengths within [band_min - window, band_min) ∪ (band_max, band_max + window],
        excluding the band interior.

        Returns
        -------
        Tensor
            Bool mask [L] selecting continuum points near the band edges.
        """
        idx = torch.where(band_mask)[0]
        if idx.numel() == 0:
            return torch.zeros_like(band_mask)
        lo_idx, hi_idx = idx.min(), idx.max()
        lo_edge = wavelength_um[lo_idx].item()
        hi_edge = wavelength_um[hi_idx].item()
        cont_mask = ((wavelength_um >= (lo_edge - window)) & (wavelength_um < lo_edge)) | (
            (wavelength_um > hi_edge) & (wavelength_um <= (hi_edge + window))
        )
        cont_mask = cont_mask & (~band_mask)
        return cont_mask

    @staticmethod
    def _second_diff(x: Tensor) -> Tensor:
        """
        Second-order finite difference along last dim: y[i] = x[i+1] - 2*x[i] + x[i-1]
        """
        return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]

    @staticmethod
    def _normalized_corr(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        """
        1 - cosine similarity (penalty) between a and b along last dimension.
        """
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        num = (a * b).sum(dim=-1)
        den = torch.sqrt((a * a).sum(dim=-1) * (b * b).sum(dim=-1) + eps)
        cos = num / (den + eps)
        return 1.0 - cos

    @staticmethod
    def _gaussian_kernel(center_um: float, wavelength_um: Tensor, hwhm_um: float) -> Tensor:
        """
        Normalized Gaussian kernel used as a simple Voigt core proxy.

        Parameters
        ----------
        center_um : float
            Line center (μm).
        wavelength_um : Tensor
            Wavelength grid [L] (μm).
        hwhm_um : float
            Half width at half maximum (μm).

        Returns
        -------
        Tensor
            Normalized kernel [L].
        """
        sigma = hwhm_um / math.sqrt(2.0 * math.log(2.0))  # HWHM -> σ
        g = torch.exp(-0.5 * ((wavelength_um - center_um) / (sigma + 1e-8)) ** 2)
        return g / (torch.norm(g) + 1e-8)

    def compute_molecular_prior_loss(
        self,
        mu: Tensor,
        wavelength_um: Tensor,
        reduction: str = "mean",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total molecular prior loss and components.

        Parameters
        ----------
        mu : Tensor
            Predicted spectrum μ(λ). Shape [L] or [B, L].
        wavelength_um : Tensor
            Wavelength grid [L] in μm.
        reduction : {'mean', 'sum'}
            Reduction across bands/batch.

        Returns
        -------
        total : Tensor
            Scalar loss (per batch if B>1).
        components : dict[str, Tensor]
            Loss components, detached.
        """
        assert wavelength_um.dim() == 1, "wavelength_um must be [L]"
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)  # [1, L]
        B, L = mu.shape
        masks = self.build_masks(wavelength_um)

        total_band_consistency: List[Tensor] = []
        total_smoothness: List[Tensor] = []
        total_edge: List[Tensor] = []
        total_voigt: List[Tensor] = []

        for band in self.bands:
            band_mask = masks[band.name]  # [L] boolean
            if band_mask.sum().item() < self.cfg.min_valid_points_per_band:
                continue

            band_mu = mu[:, band_mask]  # [B, Nb]
            cont_mask = self._continuum_mask(wavelength_um, band_mask, self.cfg.continuum_window_um)
            if cont_mask.sum().item() < max(1, self.cfg.min_valid_points_per_band // 2):
                cont_mask = (~band_mask)  # Fallback to global outside-band continuum
            cont_mu = mu[:, cont_mask]  # [B, Nc]

            # 1) Band-consistency penalty
            cont_level = torch.median(cont_mu, dim=1, keepdim=True).values  # [B, 1]
            if self.cfg.expect_higher_mu_in_band:
                band_consistency_violation = F.relu(cont_level - band_mu)
            else:
                band_consistency_violation = F.relu(band_mu - cont_level)
            band_consistency = band_consistency_violation.mean(dim=1)  # [B]
            total_band_consistency.append(band.weight * band_consistency)

            # 2) Smoothness penalty (within band)
            if band_mu.shape[1] >= 3:
                second = self._second_diff(band_mu)  # [B, Nb-2]
                smoothness = (second ** 2).mean(dim=1)  # [B]
                total_smoothness.append(band.weight * smoothness)

            # 3) Edge monotonicity penalty
            if band_mu.shape[1] >= 4:
                nb = band_mu.shape[1]
                left = band_mu[:, : nb // 2]
                right = band_mu[:, nb // 2 :]
                interior_ref = band_mu.median(dim=1, keepdim=True).values  # [B,1]
                desired_sign = 1.0 if self.cfg.expect_higher_mu_in_band else -1.0

                def slope(x: Tensor, y: Tensor) -> Tensor:
                    xm = x.mean(dim=1, keepdim=True)
                    ym = y.mean(dim=1, keepdim=True)
                    num = ((x - xm) * (y - ym)).sum(dim=1)
                    den = ((x - xm) ** 2).sum(dim=1) + 1e-8
                    return num / den  # [B]

                x_left = torch.linspace(0.0, 1.0, steps=left.shape[1], device=mu.device).unsqueeze(0).expand(B, -1)
                x_right = torch.linspace(0.0, 1.0, steps=right.shape[1], device=mu.device).unsqueeze(0).expand(B, -1)
                y_left = (left - interior_ref)
                y_right = (right - interior_ref)
                slope_left = slope(x_left, y_left)
                slope_right = slope(x_right, y_right)
                edge_pen = F.relu(-desired_sign * slope_left) + F.relu(desired_sign * slope_right)
                total_edge.append(band.weight * edge_pen)

            # 4) Voigt/Gauss-like guidance (optional)
            if self.cfg.voigt_like_weight > 0.0 and band.center_lines_um:
                template = torch.zeros(L, device=mu.device)
                for c in band.center_lines_um:
                    template += self._gaussian_kernel(c, wavelength_um, band.half_width_um)
                template = template * band_mask.float()
                if template.norm() > 0:
                    template = template / (template.norm() + 1e-8)
                    band_mu_centered = mu * band_mask.float().unsqueeze(0)  # [B, L]
                    band_mu_vec = band_mu_centered[:, band_mask]  # [B, Nb]
                    template_vec = template[band_mask].unsqueeze(0).expand(B, -1)  # [B, Nb]
                    voigt_pen = self._normalized_corr(band_mu_vec, template_vec)  # [B]
                    total_voigt.append(band.weight * voigt_pen)

        def _agg(lst: List[Tensor]) -> Tensor:
            if not lst:
                return torch.zeros(B, device=mu.device)
            x = torch.stack(lst, dim=0).sum(dim=0)  # sum bands -> [B]
            return x if reduction == "sum" else x.mean()

        comp_band = _agg(total_band_consistency) * self.cfg.band_consistency_weight
        comp_smooth = _agg(total_smoothness) * self.cfg.smoothness_weight
        comp_edge = _agg(total_edge) * self.cfg.edge_monotonicity_weight
        comp_voigt = _agg(total_voigt) * self.cfg.voigt_like_weight

        total = comp_band + comp_smooth + comp_edge + comp_voigt
        components = {
            "band_consistency": comp_band.detach(),
            "smoothness": comp_smooth.detach(),
            "edge_monotonicity": comp_edge.detach(),
            "voigt_like": comp_voigt.detach(),
        }
        return total, components


def default_molecular_priors() -> MolecularPriors:
    """
    Pragmatic default bands for Ariel-like NIR ranges (coarse; tune to instrument RSR).
      - H2O: ~1.4 μm, 1.9 μm (broad), 2.7 μm band
      - CO2: ~2.0 μm shoulder, 4.3 μm strong band (if within grid)
      - CH4: ~3.3 μm band
    """
    bands: List[MolecularBand] = [
        MolecularBand(
            name="H2O_1p4",
            windows_um=[(1.35, 1.48)],
            weight=1.0,
            center_lines_um=[1.38, 1.41, 1.44],
            half_width_um=0.01,
        ),
        MolecularBand(
            name="H2O_1p9",
            windows_um=[(1.82, 1.98)],
            weight=1.0,
            center_lines_um=[1.87, 1.92, 1.95],
            half_width_um=0.015,
        ),
        MolecularBand(
            name="H2O_2p7",
            windows_um=[(2.60, 2.80)],
            weight=0.8,
            center_lines_um=[2.65, 2.72, 2.78],
            half_width_um=0.02,
        ),
        MolecularBand(
            name="CO2_2p0",
            windows_um=[(1.98, 2.06)],
            weight=0.8,
            center_lines_um=[2.01, 2.04],
            half_width_um=0.01,
        ),
        MolecularBand(
            name="CO2_4p3",
            windows_um=[(4.20, 4.40)],
            weight=1.2,
            center_lines_um=[4.25, 4.30, 4.35],
            half_width_um=0.02,
        ),
        MolecularBand(
            name="CH4_3p3",
            windows_um=[(3.25, 3.38)],
            weight=1.0,
            center_lines_um=[3.28, 3.31, 3.35],
            half_width_um=0.015,
        ),
    ]
    return MolecularPriors(bands=bands, cfg=MolecularPrioriConfig())
