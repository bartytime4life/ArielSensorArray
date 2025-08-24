# airs_gnn.py
# SpectraMind V50 — AIRS spectral GNN head
# (c) 2025 SpectraMind Project
# License: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Graph utilities (no external deps)
# ----------------------------

def build_wavelength_graph(
    wavelengths: torch.Tensor,
    k_adj: int = 1,
    molecular_bands: Optional[Dict[str, List[int]]] = None,
    detector_groups: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an undirected graph adjacency (edge_index) for spectral channels.

    Args:
        wavelengths: (C,) sorted or unsorted wavelengths in microns or nm.
        k_adj: connect each channel to +/-k adjacent channels after sorting by wavelength.
        molecular_bands: optional mapping {"H2O": [idx...], "CO2": [...], ...} to fully connect band peers.
        detector_groups: optional groups of indices measured on the same detector region (to model correlated noise).

    Returns:
        edge_index: (2, E) LongTensor of undirected edges (i,j) (both directions included).
        edge_weight: (E,) FloatTensor with edge weights (default 1.0).
    """
    device = wavelengths.device
    C = wavelengths.numel()
    # sort by wavelength
    sort_idx = torch.argsort(wavelengths)
    inv_sort = torch.empty_like(sort_idx)
    inv_sort[sort_idx] = torch.arange(C, device=device)

    # adjacency by k-nearest in wavelength order
    edges = set()
    for rpos, i in enumerate(sort_idx.tolist()):
        for dk in range(1, k_adj + 1):
            if rpos - dk >= 0:
                j = sort_idx[rpos - dk].item()
                if i != j:
                    edges.add((min(i, j), max(i, j)))
            if rpos + dk < C:
                j = sort_idx[rpos + dk].item()
                if i != j:
                    edges.add((min(i, j), max(i, j)))

    # connect molecular band peers (complete subgraph per band)
    if molecular_bands:
        for _, idxs in molecular_bands.items():
            idxs = list(set([int(x) for x in idxs if 0 <= int(x) < C]))
            for a in idxs:
                for b in idxs:
                    if a < b:
                        edges.add((a, b))

    # optional detector region links
    if detector_groups:
        for grp in detector_groups:
            idxs = list(set([int(x) for x in grp if 0 <= int(x) < C]))
            for a in idxs:
                for b in idxs:
                    if a < b:
                        edges.add((a, b))

    if len(edges) == 0:
        # fallback: chain
        for i in range(C - 1):
            edges.add((i, i + 1))

    # build edge_index both directions
    e = list(edges)
    i0 = torch.tensor([x[0] for x in e], dtype=torch.long, device=device)
    i1 = torch.tensor([x[1] for x in e], dtype=torch.long, device=device)
    edge_index = torch.stack([torch.cat([i0, i1]), torch.cat([i1, i0])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    return edge_index, edge_weight


def normalize_adjacency(edge_index: torch.Tensor, num_nodes: int, edge_weight: Optional[torch.Tensor] = None):
    """
    Compute D^-1/2 (A + I) D^-1/2 normalized adjacency for simple GCN message passing.
    Returns indices and values for a sparse tensor.
    """
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device)

    # add self loops
    self_loops = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    ei = torch.cat([edge_index, self_loops], dim=1)
    ew = torch.cat([edge_weight, torch.ones(num_nodes, device=device)], dim=0)

    # degree
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, ei[0], ew)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    norm = deg_inv_sqrt[ei[0]] * ew * deg_inv_sqrt[ei[1]]
    return ei, norm


# ----------------------------
# Layers
# ----------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.residual = residual and (in_dim == out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ei: torch.Tensor, ew: torch.Tensor):
        """
        x: (B, C, F)
        ei: (2, E)
        ew: (E,)
        """
        B, C, F = x.shape
        # message passing: A_hat X W
        h = self.lin(x)  # (B, C, out)
        # aggregate with normalized adjacency using scatter
        row, col = ei  # messages from col -> row
        # flatten batch into node axis via offset trick
        offsets = (torch.arange(B, device=x.device) * C).view(B, 1)
        row_b = row.unsqueeze(0) + offsets
        col_b = col.unsqueeze(0) + offsets
        row_b = row_b.reshape(-1)
        col_b = col_b.reshape(-1)
        ew_b = ew.repeat(B)

        h_flat = h.reshape(B * C, -1)  # (B*C, out)
        m = h_flat[col_b] * ew_b.unsqueeze(-1)  # weighted messages
        out = torch.zeros_like(h_flat)
        out.index_add_(0, row_b, m)
        out = out.reshape(B, C, -1)

        if self.residual:
            out = out + x  # residual if dims match
        out = self.ln(out)
        out = F.relu(out)
        out = self.drop(out)
        return out


class NodeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Model
# ----------------------------

@dataclass
class AirsGNNConfig:
    in_features: int = 8         # per-channel feature count
    hidden: int = 128
    gcn_layers: int = 4
    gcn_dropout: float = 0.1
    mlp_hidden: int = 128
    mlp_dropout: float = 0.1
    k_adj: int = 1
    smooth_lambda: float = 1e-3
    nonneg_lambda: float = 1e-3
    group_lambda: float = 5e-4
    min_log_sigma: float = -7.0
    max_log_sigma: float = 3.0


class AirsSpectralGNN(nn.Module):
    """
    Spectral GNN over AIRS wavelength channels.

    Inputs:
        x:  (B, C, F) per-channel features (e.g., calibrated flux stats, template matches, etc.)
        wavelengths: (C,) wavelengths (float)
        molecular_bands: optional dict molecule->list of indices to enforce relational edges.

    Outputs:
        dict with:
            mu:         (B, C) predicted mean spectrum
            log_sigma:  (B, C) predicted log std
    """
    def __init__(self, C: int, cfg: AirsGNNConfig):
        super().__init__()
        self.C = C
        self.cfg = cfg

        self.input_proj = NodeMLP(cfg.in_features, cfg.mlp_hidden, cfg.hidden, cfg.mlp_dropout)

        gcn = []
        for _ in range(cfg.gcn_layers):
            gcn.append(GCNLayer(cfg.hidden, cfg.hidden, dropout=cfg.gcn_dropout, residual=True))
        self.gcn = nn.ModuleList(gcn)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 2),  # (mu, log_sigma)
        )

    def forward(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        molecular_bands: Optional[Dict[str, List[int]]] = None,
        detector_groups: Optional[List[List[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, F = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        # Build graph (once per forward; caller can cache if desired)
        edge_index, edge_weight = build_wavelength_graph(
            wavelengths=wavelengths,
            k_adj=self.cfg.k_adj,
            molecular_bands=molecular_bands,
            detector_groups=detector_groups,
        )
        ei, ew = normalize_adjacency(edge_index, num_nodes=C, edge_weight=edge_weight)

        h = self.input_proj(x)
        for layer in self.gcn:
            h = layer(h, ei, ew)

        out = self.head(h)  # (B, C, 2)
        mu = out[..., 0]
        log_sigma = out[..., 1].clamp(self.cfg.min_log_sigma, self.cfg.max_log_sigma)
        return {"mu": mu, "log_sigma": log_sigma, "edge_index": ei, "edge_weight": ew}


# ----------------------------
# Loss (Gaussian NLL + physics penalties)
# ----------------------------

class AirsGNNLoss(nn.Module):
    def __init__(self, cfg: AirsGNNConfig, molecular_bands: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.cfg = cfg
        self.molecular_bands = molecular_bands or {}

    @staticmethod
    def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Per‑node heteroscedastic Gaussian NLL.
        y, mu: (B, C); log_sigma: (B, C)
        mask: (B, C) bool or 0/1
        """
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (math.log(2 * math.pi) + 2.0 * log_sigma + (y - mu) ** 2 / var)
        if mask is not None:
            nll = nll * mask
            denom = mask.sum().clamp(min=1.0)
        else:
            denom = torch.tensor(y.numel(), device=y.device, dtype=y.dtype)
        return nll.sum() / denom

    def smoothness_penalty(self, mu: torch.Tensor, ei: torch.Tensor, ew: torch.Tensor):
        """
        Graph Laplacian smoothness: sum_{(i,j) in E} w_ij (mu_i - mu_j)^2
        mu: (B, C)
        """
        row, col = ei
        diff = mu[:, row] - mu[:, col]
        # Each undirected edge appears twice (i->j, j->i); halve to avoid double count
        return (ew * (diff ** 2)).sum(dim=1).mean() * 0.5

    def nonneg_penalty(self, mu: torch.Tensor):
        # Encourage non-negative transmission depths (or clamp to physical bounds if normalized):
        return torch.relu(-mu).mean()

    def group_consistency_penalty(self, mu: torch.Tensor):
        """
        Encourage molecules’ linked lines to move together:
        sum_bands mean_i (mu_i - band_mean)^2
        """
        if not self.molecular_bands:
            return mu.new_tensor(0.0)
        B, C = mu.shape
        total = mu.new_tensor(0.0)
        bands = 0
        for _, idxs in self.molecular_bands.items():
            idx = [i for i in idxs if 0 <= i < C]
            if len(idx) < 2:
                continue
            bands += 1
            group = mu[:, idx]  # (B, K)
            gmean = group.mean(dim=1, keepdim=True)
            total = total + ((group - gmean) ** 2).mean()
        if bands == 0:
            return mu.new_tensor(0.0)
        return total / bands

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        mu, log_sigma = preds["mu"], preds["log_sigma"]
        ei, ew = preds["edge_index"], preds["edge_weight"]

        nll = self.gaussian_nll(mu, log_sigma, target, mask=mask)
        smooth = self.smoothness_penalty(mu, ei, ew) * self.cfg.smooth_lambda
        nonneg = self.nonneg_penalty(mu) * self.cfg.nonneg_lambda
        groupc = self.group_consistency_penalty(mu) * self.cfg.group_lambda
        total = nll + smooth + nonneg + groupc

        components = {
            "loss_total": total.detach(),
            "loss_nll": nll.detach(),
            "loss_smooth": smooth.detach(),
            "loss_nonneg": nonneg.detach(),
            "loss_group": groupc.detach(),
        }
        return total, components


# ----------------------------
# Tiny usage example / self-test
# ----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, F = 2, 283, 8
    wl = torch.linspace(0.5, 1.8, C)  # microns
    x = torch.randn(B, C, F)

    # Example molecular bands (dummy indices)
    mol = {
        "H2O": [i for i in range(40, 70, 2)],
        "CO2": [i for i in range(120, 150, 3)],
        "CH4": [i for i in range(200, 235, 4)],
    }

    cfg = AirsGNNConfig()
    model = AirsSpectralGNN(C=C, cfg=cfg)
    out = model(x, wavelengths=wl, molecular_bands=mol)

    # fake target spectrum and mask
    y = torch.randn(B, C) * 0.05 + 0.2  # “mostly positive”
    mask = torch.ones(B, C)

    criterion = AirsGNNLoss(cfg=cfg, molecular_bands=mol)
    loss, parts = criterion(out, y, mask)
    print("Loss:", float(loss))
    print(parts)