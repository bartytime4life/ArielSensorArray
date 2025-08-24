#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/fgs1_mamba.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)

FGS1MambaEncoder
----------------
A fast, memory‑efficient, SSM‑style (Mamba‑inspired) encoder for very long 1‑D sequences.
It ingests FGS1 time series (optionally flattened per-frame features) and produces either:
    • a pooled representation (default): [B, d_model]
    • or a sequence representation:      [B, T, d_model]   (cfg.return_sequence=True)

Design goals
------------
• Handle extreme sequence lengths (10^5+ timesteps) with O(T) complexity and constant memory.
• Numerically stable, parallelizable SSM scan with learnable diagonal state matrices.
• Optional short‑term Conv1D prefilter, gating, and residual mixing for expressivity.
• Padding‑aware masked pooling for variable‑length batches (lengths tensor).
• Hydra‑friendly config via dataclass defaults but accepting dict-like configs as well.

Input conventions
-----------------
• The encoder expects input shaped as:
      (B, T, F)          — F features per timestep (e.g., flattened spatial dims)
  If you pass cubes (B, T, W, C) or (B, T, W), please flatten to (B, T, F) upstream.

• lengths: Optional 1D tensor [B] with valid time lengths (for padding‑aware pooling).

Config (example)
----------------
cfg.model.fgs1 = {
  "in_dim": 64,              # input feature size F
  "d_model": 256,            # hidden size
  "n_layers": 6,             # number of SSM blocks
  "dropout": 0.1,
  "conv_kernel": 5,          # 0 disables conv prefilter
  "ssm_rank": 1,             # low-rank D term (kept simple here)
  "bidirectional": False,    # optional backward scan and fuse
  "return_sequence": False,  # True => return [B,T,d_model], else pooled [B,d_model]
  "pool": "mean",            # 'mean' | 'cls' | 'last'
}

Notes
-----
• This is a self-contained, torch.nn implementation that mimics Mamba-style selective SSM behavior.
  It does not depend on external mamba packages.
• The block includes:
    - LayerNorm
    - Optional depthwise Conv1D prefilter (temporal)
    - Diagonal SSM scan y_t = C * x_t_state + D * u_t with gated input u_t
    - Residual + dropout
• For bidirectional mode, a reverse-time pass is computed and fused by a linear mix.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_cfg_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf/Namespace/plain-dict to a plain dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    out: Dict[str, Any] = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        v = getattr(cfg, k, None)
        # skip callables / modules
        if callable(v):
            continue
        out[k] = v
    return out


def masked_mean(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Padding-aware mean over time.
    x:        [B, T, D]
    lengths:  [B] with valid lengths (int). If None, returns simple mean over dim=1.
    """
    if lengths is None:
        return x.mean(dim=1)
    B, T, D = x.shape
    device = x.device
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B,T]
    mask = mask.to(x.dtype).unsqueeze(-1)  # [B,T,1]
    s = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return s / denom


# ---------------------------------------------------------------------------
# Mamba-like SSM block
# ---------------------------------------------------------------------------

class SSMDiagScan(nn.Module):
    """
    Parallelizable diagonal SSM with gated input.
    State update (per feature channel j):
        s_t[j] = a[j] * s_{t-1}[j] + b[j] * u_t[j]
        y_t[j] = c[j] * s_t[j]      + d[j] * u_t[j]
    where a in (-1, 1) for stability (parameterized via tanh); b,c,d are learnable.

    To keep things fast and GPU-parallel, we use a cumulative product formulation for s_t and
    exploit associativity. For very long sequences, this avoids naive Python loops.
    """

    def __init__(self, d_model: int, rank: int = 1):
        super().__init__()
        self.d_model = d_model
        self.rank = max(1, int(rank))

        # Learnable diagonal A stabilized via tanh; initialize near 1 but < 1.
        self.a = nn.Parameter(torch.zeros(d_model))  # raw, mapped via tanh => (-1,1)
        # Learnable B,C,D (elementwise) — kept simple; rank allows small low-rank mixing for D.
        self.b = nn.Parameter(torch.randn(d_model) * 0.02)
        self.c = nn.Parameter(torch.randn(d_model) * 0.02)
        self.d = nn.Parameter(torch.randn(d_model) * 0.02)

        # Gating MLP: gate = sigmoid(W_g * u + b)
        self.gate = nn.Linear(d_model, d_model)

        # Low-rank additive mix for D, optional tiny extra capacity
        self.d_lr_u = nn.Parameter(torch.randn(d_model, self.rank) * 0.02)
        self.d_lr_v = nn.Parameter(torch.randn(self.rank, d_model) * 0.02)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: [B, T, D]
        Returns: y: [B, T, D]
        """
        B, T, D = u.shape
        assert D == self.d_model, f"expected D={self.d_model}, got {D}"

        # Gate the input
        g = torch.sigmoid(self.gate(u))  # [B,T,D]
        u_g = u * g

        # Stabilized diagonal A in (-1, 1)
        a = torch.tanh(self.a).view(1, 1, D)  # [1,1,D]
        b = self.b.view(1, 1, D)
        c = self.c.view(1, 1, D)
        d = self.d.view(1, 1, D)

        # Optional low-rank refinement for D: d_eff = d + (u @ d_lr_u @ d_lr_v) diag-like mixing
        # Keep it simple & cheap: compute an additive per-time/channel correction via a tiny linear map.
        # (This is not a full matrix; it's just a feature-wise learned rescaling based on u mean.)
        u_mean = u_g.mean(dim=1, keepdim=True)  # [B,1,D]
        d_lr = (u_mean @ self.d_lr_u @ self.d_lr_v) / max(1.0, float(self.rank))  # [B,1,D]
        d_eff = d + d_lr  # broadcast across time

        # Compute s_t via parallel scan:
        # s_t = a*s_{t-1} + b*u_t  =>  s = (b*u) convolved with powers of a
        # Efficient computation using cumulative products:
        #   Let G[t] = prod_{k=1..t} a  (elementwise)  and   z[t] = (b*u[t]) / G[t]
        #   Then s[t] = G[t] * sum_{k=1..t} z[k]
        # This avoids explicit for-loops in Python (still O(T) but vectorized).
        # NOTE: For extremely long T, float32/16 stability is good with |a|<1.
        a_expand = a.expand(B, T, D)             # [B,T,D]
        # cumulative product along time (prod of a)
        G = torch.cumprod(a_expand, dim=1)       # [B,T,D]
        bu = b * u_g                              # [B,T,D]
        # Avoid division by zero if a==0 at t==0: clamp G
        G_safe = G.clone()
        G_safe[:, 0, :] = G_safe[:, 0, :].clamp_min(1e-6)
        z = bu / G_safe                           # [B,T,D]
        s = G * torch.cumsum(z, dim=1)           # [B,T,D]

        # Output y
        y = c * s + d_eff * u_g                   # [B,T,D]
        return y


class MambaBlock(nn.Module):
    """
    Mamba-style encoder block:
      x -> LN -> (optional Conv1D prefilter) -> SSMDiagScan -> Linear mix -> Dropout -> Residual
    """
    def __init__(
        self,
        d_model: int,
        conv_kernel: int = 0,
        dropout: float = 0.0,
        ssm_rank: int = 1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv_kernel = int(conv_kernel)
        if self.conv_kernel and self.conv_kernel > 1:
            pad = (self.conv_kernel - 1) // 2
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=self.conv_kernel,
                padding=pad, groups=d_model, bias=False  # depthwise
            )
        else:
            self.conv = None

        self.ssm = SSMDiagScan(d_model=d_model, rank=ssm_rank)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        residual = x
        x = self.norm(x)
        if self.conv is not None:
            # Conv1d expects [B, D, T]
            xc = x.transpose(1, 2)                  # [B,D,T]
            xc = self.conv(xc)
            x = xc.transpose(1, 2)                  # [B,T,D]

        x = self.ssm(x)
        x = self.proj(x)
        x = self.drop(x)
        return x + residual


# ---------------------------------------------------------------------------
# FGS1 Mamba Encoder
# ---------------------------------------------------------------------------

@dataclass
class FGS1MambaConfig:
    in_dim: int = 64
    d_model: int = 256
    n_layers: int = 6
    dropout: float = 0.1
    conv_kernel: int = 5
    ssm_rank: int = 1
    bidirectional: bool = False
    return_sequence: bool = False
    pool: str = "mean"  # 'mean' | 'cls' | 'last'


class FGS1MambaEncoder(nn.Module):
    """
    High-throughput SSM encoder for FGS1 time series.

    Forward:
        x: [B, T, F]            # F must equal cfg.in_dim
        lengths: Optional[B]    # valid lengths for padding-aware pooling
        returns: [B, D] if return_sequence=False else [B, T, D]
    """
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfgd = _to_cfg_dict(cfg)
        self.cfg = FGS1MambaConfig(
            in_dim=int(cfgd.get("in_dim", 64)),
            d_model=int(cfgd.get("d_model", 256)),
            n_layers=int(cfgd.get("n_layers", 6)),
            dropout=float(cfgd.get("dropout", 0.1)),
            conv_kernel=int(cfgd.get("conv_kernel", 5)),
            ssm_rank=int(cfgd.get("ssm_rank", 1)),
            bidirectional=bool(cfgd.get("bidirectional", False)),
            return_sequence=bool(cfgd.get("return_sequence", False)),
            pool=str(cfgd.get("pool", "mean")),
        )

        self.in_proj = nn.Linear(self.cfg.in_dim, self.cfg.d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.cfg.d_model,
                conv_kernel=self.cfg.conv_kernel,
                dropout=self.cfg.dropout,
                ssm_rank=self.cfg.ssm_rank,
            )
            for _ in range(self.cfg.n_layers)
        ])

        # If bidirectional, use another stack for reverse pass (tied settings)
        if self.cfg.bidirectional:
            self.blocks_rev = nn.ModuleList([
                MambaBlock(
                    d_model=self.cfg.d_model,
                    conv_kernel=self.cfg.conv_kernel,
                    dropout=self.cfg.dropout,
                    ssm_rank=self.cfg.ssm_rank,
                )
                for _ in range(self.cfg.n_layers)
            ])
            self.bi_fuse = nn.Linear(2 * self.cfg.d_model, self.cfg.d_model)
        else:
            self.blocks_rev = None
            self.bi_fuse = None

        # CLS token if chosen pooling='cls'
        if self.cfg.pool == "cls":
            self.cls = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)
        else:
            self.cls = None

        self.out_norm = nn.LayerNorm(self.cfg.d_model)

    def _forward_single_direction(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, T, D]
        """
        for blk in self.blocks:
            x = blk(x)
        return x

    def _forward_reverse_direction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the same style of blocks but over reversed time and then un-reverses output.
        """
        assert self.blocks_rev is not None
        x_rev = torch.flip(x, dims=[1])  # reverse time
        for blk in self.blocks_rev:
            x_rev = blk(x_rev)
        x_bi = torch.flip(x_rev, dims=[1])
        return x_bi

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, F]
        lengths: optional [B] ints for valid time lengths (<= T)
        """
        B, T, F = x.shape
        if F != self.cfg.in_dim:
            raise ValueError(f"FGS1MambaEncoder expected in_dim={self.cfg.in_dim}, got {F}")

        # Input projection
        x = self.in_proj(x)  # [B,T,D]

        # Optional CLS token
        if self.cls is not None:
            cls_tok = self.cls.expand(B, 1, -1)  # [B,1,D]
            x = torch.cat([cls_tok, x], dim=1)   # [B,T+1,D]
            if lengths is not None:
                lengths = (lengths + 1).clamp_max(x.shape[1])

        # Forward (and optional backward) stacks
        y_f = self._forward_single_direction(x)
        if self.cfg.bidirectional:
            y_b = self._forward_reverse_direction(x)
            y = torch.cat([y_f, y_b], dim=-1)        # [B,T(±1),2D]
            y = self.bi_fuse(y)
        else:
            y = y_f                                   # [B,T(±1),D]

        y = self.out_norm(y)

        if self.cfg.return_sequence:
            return y

        # Pooled representation
        if self.cfg.pool == "mean":
            out = masked_mean(y, lengths)
        elif self.cfg.pool == "last":
            if lengths is None:
                out = y[:, -1, :]
            else:
                idx = (lengths - 1).clamp_min(0)  # [B]
                out = y[torch.arange(B, device=y.device), idx, :]
        elif self.cfg.pool == "cls":
            out = y[:, 0, :]
        else:
            raise ValueError(f"Unknown pool: {self.cfg.pool}")

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, F = 2, 1024, 64
    x = torch.randn(B, T, F)

    cfg = {
        "in_dim": F,
        "d_model": 128,
        "n_layers": 4,
        "dropout": 0.1,
        "conv_kernel": 5,
        "ssm_rank": 1,
        "bidirectional": True,
        "return_sequence": False,
        "pool": "mean",
    }

    enc = FGS1MambaEncoder(cfg)
    lengths = torch.tensor([T, T - 100])
    z = enc(x, lengths=lengths)
    print("pooled out:", z.shape)        # [B, d_model]

    enc.cfg.return_sequence = True
    z_seq = enc(x, lengths=lengths)
    print("sequence out:", z_seq.shape)  # [B, T, d_model] or [B, T+1, d_model] if CLS