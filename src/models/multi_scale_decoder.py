# multi_scale_decoder.py
# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
# (c) 2025 SpectraMind Team. MIT License.
"""
Multi-scale decoder head that:
  • Fuses pyramid/ multi-resolution features from the encoder
  • Predicts per-wavelength mean and variance (heteroscedastic Gaussian)
  • Optionally applies graph-Laplacian spectral refinement
  • Provides physics-aware constraints (non-negativity, range clamps)
  • Exposes helper losses (GLL / quantile) and smoothness regularizers

Design notes
------------
- Fusion: lightweight cross-scale attention to weight each scale, plus residual MLP
- Outputs: 'gaussian' (mu, log_var) for GLL metric; or 'quantile' (τ∈{.1,.5,.9})
- Refinement: Laplacian smoothing on the 283-length spectrum; works with or without
  torch_geometric. Encodes smoothness & bandwise correlation priors.
- Physics constraints: non-negative transit depth, optional [0,1] clamp
- Repo integration: pure PyTorch; Hydra-ready via dataclass config

References (context)
--------------------
• V50 pipeline favors a hybrid model stack and physics-informed overlays (rules/priors). 
• The challenge metric uses a Gaussian log-likelihood with penalties for overconfidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

try:
    # Optional acceleration when available
    from torch_geometric.nn import MessagePassing  # type: ignore
    _HAS_PYG = True
except Exception:  # pragma: no cover
    _HAS_PYG = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class DecoderConfig:
    num_wavelengths: int = 283
    in_dims: Tuple[int, ...] = (256, 256, 256)  # one per scale
    hidden: int = 512
    dropout: float = 0.1
    out_mode: str = "gaussian"  # 'gaussian' or 'quantile'
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    temporal_pool: str = "attn"  # 'mean' | 'max' | 'attn'
    use_laplacian_refine: bool = True
    laplacian_alpha: float = 0.1
    laplacian_iters: int = 3
    laplacian_lambda: float = 1e-3  # training-time smoothness penalty
    physics_nonneg: bool = True
    physics_sigmoid_range: Optional[Tuple[float, float]] = None  # e.g. (0.0, 1.0)


# ----------------------------
# Utilities
# ----------------------------
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    # Accept [B, C] or [B, C, T]; pool time if present
    if x.dim() == 3:
        return x.mean(-1)
    elif x.dim() == 2:
        return x
    else:
        raise ValueError(f"Expected [B,C] or [B,C,T], got {tuple(x.shape)}")


def build_spectral_laplacian(
    num_wavelengths: int,
    connect_adjacent: bool = True,
    extra_band_edges: Optional[Sequence[Tuple[int, int]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct a simple unnormalized Laplacian L = D - A for the 1D spectrum graph.
    - Adjacent channel connections
    - Optional extra fully-connected band cliques via 'extra_band_edges' (inclusive bounds)
    """
    N = num_wavelengths
    A = torch.zeros((N, N), dtype=torch.float32, device=device)
    if connect_adjacent:
        i = torch.arange(N - 1, device=device)
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    if extra_band_edges:
        for lo, hi in extra_band_edges:
            lo = max(0, int(lo))
            hi = min(N - 1, int(hi))
            if hi > lo:
                rng = torch.arange(lo, hi + 1, device=device)
                A[rng.unsqueeze(1), rng.unsqueeze(0)] = 1.0
                A[rng, rng] = 0.0  # No self loops
    D = torch.diag(A.sum(dim=1))
    L = D - A
    return L


def laplacian_refine(
    y: torch.Tensor,  # [B, N]
    L: torch.Tensor,  # [N, N]
    alpha: float = 0.1,
    iters: int = 3,
) -> torch.Tensor:
    """
    Iterative Jacobi-like refinement: y <- y - alpha * y L
    (Neumann approx of (I + alpha L)^{-1} smoothing)
    """
    # y: [B,N], L: [N,N]
    for _ in range(iters):
        y = y - alpha * (y @ L)
    return y


# ----------------------------
# Modules
# ----------------------------
class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin2(self.drop(self.act(self.lin1(self.norm(x)))))
        return x + self.drop(h)


class TemporalPool(nn.Module):
    """Pool [B,C,T] -> [B,C] with mean/max or learned attention."""

    def __init__(self, mode: str, in_dim: int):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.q = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if self.mode == "mean":
            return x.mean(-1)
        if self.mode == "max":
            return x.max(-1).values
        if self.mode == "attn":
            # scores over time
            w = self.q(x.transpose(1, 2)).squeeze(-1)  # [B,T,1] -> [B,T]
            w = w.softmax(dim=-1)
            return (x * w.unsqueeze(1)).sum(-1)
        raise ValueError(f"Unknown pool mode: {self.mode}")


class CrossScaleAttention(nn.Module):
    """
    Given K scale embeddings (all mapped to same hidden), compute per-scale weights and fuse.
    """

    def __init__(self, hidden: int, num_scales: int, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.alpha = nn.Parameter(torch.zeros(num_scales))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, scales: List[torch.Tensor]) -> torch.Tensor:
        """
        scales: list of [B,H]
        fuse: softmax over scales from query of last scale against keys
        """
        assert len(scales) == self.num_scales
        H = scales[-1]
        K = torch.stack(scales, dim=1)  # [B,K,H]
        q = self.query(self.norm(H)).unsqueeze(1)  # [B,1,H]
        k = self.key(self.norm(K))  # [B,K,H]
        logits = (q * k).sum(-1) + self.alpha  # [B,K]
        w = logits.softmax(dim=1)  # [B,K]
        fused = (K * w.unsqueeze(-1)).sum(1)  # [B,H]
        return self.drop(fused)


# Optional PyG-based one-hop refiner (used only if torch_geometric is there)
class _PyGLaplacianRefiner(nn.Module):  # pragma: no cover
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, mu: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # mu: [B,N]; edge_index: [2,E]
        # Simple neighbor averaging + MLP
        row, col = edge_index
        B, N = mu.shape
        # aggregate neighbor mean per node
        out = torch.zeros_like(mu)
        deg = torch.zeros((N,), device=mu.device).scatter_add_(0, row, torch.ones_like(row, dtype=mu.dtype))
        deg = deg.clamp(min=1.0)
        # For each batch independently
        for b in range(B):
            m = mu[b]  # [N]
            agg = torch.zeros_like(m)
            agg.scatter_add_(0, row, m[col])
            out[b] = agg / deg
        return self.mlp(out.unsqueeze(-1)).squeeze(-1)


# ----------------------------
# Decoder
# ----------------------------
class MultiScaleDecoder(nn.Module):
    """
    Multi-scale fusion decoder for spectral prediction.

    Inputs
    ------
    features: List/Tuple/Dict of tensors per scale. Each tensor can be [B,C] or [B,C,T].
              If Dict, values are taken in insertion/key order (Python 3.7+ preserves order).

    Outputs
    -------
    mode='gaussian': returns dict(mu=[B,N], log_var=[B,N])
    mode='quantile': returns dict(q=[B,N,Q], quantiles=tuple)
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        K = len(cfg.in_dims)
        self.pools = nn.ModuleList([TemporalPool(cfg.temporal_pool, d) for d in cfg.in_dims])
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, cfg.hidden), nn.GELU()) for d in cfg.in_dims])
        self.fuse = CrossScaleAttention(cfg.hidden, num_scales=K, dropout=cfg.dropout)
        self.trunk = nn.Sequential(
            ResidualMLP(cfg.hidden, cfg.dropout),
            ResidualMLP(cfg.hidden, cfg.dropout),
        )
        out_dim = cfg.num_wavelengths if cfg.out_mode == "gaussian" else cfg.num_wavelengths * len(cfg.quantiles)
        self.head_mu = nn.Linear(cfg.hidden, cfg.num_wavelengths)
        self.head_lv = nn.Linear(cfg.hidden, cfg.num_wavelengths) if cfg.out_mode == "gaussian" else None
        self.head_q = nn.Linear(cfg.hidden, out_dim) if cfg.out_mode == "quantile" else None

        # Spectral Laplacian built on first use (lazy) unless provided externally
        self.register_buffer("L", torch.empty(0), persistent=False)
        self.pyglayer = _PyGLaplacianRefiner(cfg.hidden) if _HAS_PYG and cfg.use_laplacian_refine else None

    def _gather_scales(self, features: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        if isinstance(features, dict):
            feats = list(features.values())
        else:
            feats = list(features)
        assert len(feats) == len(self.cfg.in_dims), f"Expected {len(self.cfg.in_dims)} scales, got {len(feats)}"
        outs: List[torch.Tensor] = []
        for f, pool, proj in zip(feats, self.pools, self.proj):
            f2 = pool(f)  # [B,C]
            outs.append(proj(f2))  # [B,H]
        return outs

    def forward(
        self,
        features: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        laplacian: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scales = self._gather_scales(features)  # list of [B,H]
        z = self.trunk(self.fuse(scales))  # [B,H]

        if self.cfg.out_mode == "gaussian":
            mu = self.head_mu(z)  # [B,N]
            log_var = self.head_lv(z)  # [B,N]
            # Physics-aware constraints
            if self.cfg.physics_nonneg:
                mu = F.softplus(mu)  # ≥0
            if self.cfg.physics_sigmoid_range is not None:
                lo, hi = self.cfg.physics_sigmoid_range
                mu = torch.sigmoid(mu) * (hi - lo) + lo

            # Optional refinement
            if self.cfg.use_laplacian_refine:
                L = laplacian if laplacian is not None else self._get_or_build_L(mu.device)
                if self.pyglayer is not None and edge_index is not None:
                    mu = mu + self.pyglayer(mu, edge_index)  # residual refinement
                else:
                    mu = laplacian_refine(mu, L, self.cfg.laplacian_alpha, self.cfg.laplacian_iters)
            return {"mu": mu, "log_var": log_var}

        elif self.cfg.out_mode == "quantile":
            q = self.head_q(z)  # [B, N*Q]
            q = q.view(z.size(0), self.cfg.num_wavelengths, len(self.cfg.quantiles))
            if self.cfg.physics_nonneg:
                q = F.softplus(q)
            if self.cfg.physics_sigmoid_range is not None:
                lo, hi = self.cfg.physics_sigmoid_range
                q = torch.sigmoid(q) * (hi - lo) + lo
            if self.cfg.use_laplacian_refine:
                L = laplacian if laplacian is not None else self._get_or_build_L(q.device)
                # refine median only to preserve quantile ordering
                mid_idx = self.cfg.quantiles.index(0.5) if 0.5 in self.cfg.quantiles else len(self.cfg.quantiles) // 2
                qm = q[..., mid_idx]
                qm = laplacian_refine(qm, L, self.cfg.laplacian_alpha, self.cfg.laplacian_iters)
                q = q.clone()
                q[..., mid_idx] = qm
            return {"q": q, "quantiles": torch.tensor(self.cfg.quantiles, device=q.device)}
        else:
            raise ValueError(f"Unsupported out_mode={self.cfg.out_mode}")

    def _get_or_build_L(self, device: torch.device) -> torch.Tensor:
        if self.L.numel() == 0:
            self.L = build_spectral_laplacian(self.cfg.num_wavelengths, device=device)
        return self.L


# ----------------------------
# Losses / Regularizers
# ----------------------------
def gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood (per element):
      0.5 * [ log(2π) + log_var + (y - mu)^2 / exp(log_var) ]
    """
    var = (log_var).exp().clamp_min(1e-9)
    loss = 0.5 * (torch.log(2 * torch.pi) + log_var + (y - mu) ** 2 / var)
    return loss.mean() if reduction == "mean" else loss.sum()


def quantile_loss(q: torch.Tensor, y: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """
    Pinball loss for multiple quantiles. q: [B,N,Q], y: [B,N]
    """
    diffs = y.unsqueeze(-1) - q  # positive if y>q
    losses = []
    for i, tau in enumerate(quantiles):
        ei = diffs[..., i]
        losses.append(torch.maximum(tau * ei, (tau - 1) * ei))
    return torch.mean(torch.stack(losses, dim=-1))


def laplacian_smoothness(mu: torch.Tensor, L: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Smoothness prior: sum_i mu^T L mu (per batch)
    """
    s = torch.einsum("bi,ij,bj->b", mu, L, mu)
    return s.mean() if reduction == "mean" else s.sum()


def decoder_loss(
    outputs: Dict[str, torch.Tensor],
    y: torch.Tensor,
    L: Optional[torch.Tensor] = None,
    lambda_smooth: float = 1e-3,
    quantiles: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Combined decoder loss:
      - Gaussian mode: NLL + λ * smooth(mu)
      - Quantile mode: Pinball + λ * smooth(median)
    """
    if "mu" in outputs and "log_var" in outputs:
        mu, lv = outputs["mu"], outputs["log_var"]
        loss = gaussian_nll(mu, lv, y)
        if L is not None and lambda_smooth > 0:
            loss = loss + lambda_smooth * laplacian_smoothness(mu, L)
        return loss
    elif "q" in outputs and quantiles is not None:
        q = outputs["q"]
        loss = quantile_loss(q, y, quantiles)
        if L is not None and lambda_smooth > 0:
            # use median for smoothing
            mid_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            loss = loss + lambda_smooth * laplacian_smoothness(q[..., mid_idx], L)
        return loss
    else:
        raise ValueError("Outputs not recognized for loss.")


# ----------------------------
# Quick self-test
# ----------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = DecoderConfig(
        num_wavelengths=283,
        in_dims=(128, 256, 256),
        hidden=384,
        out_mode="gaussian",
        laplacian_alpha=0.05,
        laplacian_iters=2,
        laplacian_lambda=1e-3,
        physics_nonneg=True,
    )
    dec = MultiScaleDecoder(cfg)

    B, T = 4, 1024
    f1 = torch.randn(B, 128, T)  # temporal scale 1
    f2 = torch.randn(B, 256, T)  # temporal scale 2
    f3 = torch.randn(B, 256)     # already pooled

    outs = dec([f1, f2, f3])
    mu, lv = outs["mu"], outs["log_var"]
    LOGGER.info(f"mu shape={mu.shape}, log_var shape={lv.shape}")

    # A toy target and laplacian loss
    y = torch.rand_like(mu) * 0.01  # small transit depths
    L = build_spectral_laplacian(cfg.num_wavelengths, extra_band_edges=[(120, 150), (210, 230)], device=mu.device)
    loss = decoder_loss(outs, y, L=L, lambda_smooth=cfg.laplacian_lambda)
    LOGGER.info(f"loss={loss.item():.5f}")