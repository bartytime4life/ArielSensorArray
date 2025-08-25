# test_spectral_absorption_overlay_clustered.py
# -----------------------------------------------------------------------------
# SpectraMind V50 — Diagnostics
#
# Purpose:
#   Diagnostic test that synthesizes a transmission spectrum with multiple
#   molecular absorption bands, clusters nearby bands, and overlays the
#   clustered bands on the spectrum. The test asserts basic scientific and
#   rendering invariants (bounds, non-negativity, cluster counts) and writes
#   a PNG artifact for quick visual inspection in CI.
#
# Notes (contextual background for the diagnostic):
#   - Planetary transmission spectra exhibit discrete absorption features
#     (quantized line/ band structure) tied to molecular energy level
#     differences; we visualize and cluster these features for QA. 
#   - Lines/bands of the same molecule appear at consistent wavelengths; a
#     simple proximity-based clusterer helps sanity-check that our bands are
#     coherent and smooth across neighboring channels (spectral smoothness
#     prior). 
#
# This header comment is informational only; the test assertions below do not
# rely on these references.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

# Use non-interactive backend for headless CI
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ------------------------------ Utilities ------------------------------------


@dataclass(frozen=True)
class Band:
    molecule: str
    wl_min: float  # microns
    wl_max: float  # microns

    @property
    def center(self) -> float:
        return 0.5 * (self.wl_min + self.wl_max)

    @property
    def width(self) -> float:
        return self.wl_max - self.wl_min


@dataclass
class Cluster:
    molecule: str
    wl_min: float
    wl_max: float
    members: List[Band]

    @property
    def center(self) -> float:
        return 0.5 * (self.wl_min + self.wl_max)


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)


def simulate_transmission_spectrum(
    wl: np.ndarray,
    bands: Sequence[Band],
    line_depth: float = 0.02,
    continuum_level: float = 1.0,
    noise_ppm: float = 50.0,
) -> np.ndarray:
    """
    Create a synthetic transmission spectrum T(λ) with Gaussian-shaped
    absorption for each band and photon-like noise. Output is normalized
    (0, 1], with 1=continuum (no absorption), dips at lines.

    Parameters
    ----------
    wl : array of wavelengths (microns)
    bands : list of absorption bands (with [wl_min, wl_max])
    line_depth : typical depth of individual line contributions
    continuum_level : base level (≈1.0 for transmission)
    noise_ppm : approximate white noise level in ppm

    Returns
    -------
    T : np.ndarray
        Transmission spectrum ∈ (0, 1]
    """
    T = np.full_like(wl, fill_value=continuum_level, dtype=float)

    # Superpose smooth band structures using sums of Gaussians per band.
    for b in bands:
        # Place 3 sub-lines within the band region to simulate substructure.
        centers = np.linspace(b.wl_min, b.wl_max, 3)
        for c in centers:
            sigma = 0.15 * (b.wl_max - b.wl_min + 1e-6)  # modest broadening
            contrib = line_depth * np.exp(-0.5 * ((wl - c) / sigma) ** 2)
            T -= contrib

    # Add small random noise resembling photon noise at ~noise_ppm.
    # Convert ppm to fractional standard deviation.
    sigma_n = noise_ppm * 1e-6
    T += np.random.normal(0.0, sigma_n, size=wl.shape)

    # Clamp to (0, 1] for physical plausibility of transmission
    T = np.clip(T, 1e-6, 1.0)

    return T


def cluster_bands_by_proximity(
    bands: Sequence[Band],
    max_gap: float = 0.08,
) -> List[Cluster]:
    """
    Cluster band intervals by merging adjacent/overlapping bands if the gap
    between them is ≤ max_gap. Clustering is done per molecule to reflect
    chemistry-specific groupings.

    Parameters
    ----------
    bands : list of Band
    max_gap : float
        Maximum allowed gap (microns) to merge bands within the same molecule.

    Returns
    -------
    clusters : list of Cluster
    """
    clusters: List[Cluster] = []
    # Group by molecule
    by_mol: Dict[str, List[Band]] = {}
    for b in bands:
        by_mol.setdefault(b.molecule, []).append(b)

    for mol, mol_bands in by_mol.items():
        # Sort by start wavelength
        mol_bands = sorted(mol_bands, key=lambda x: (x.wl_min, x.wl_max))
        current_members: List[Band] = []
        cur_min, cur_max = None, None

        for b in mol_bands:
            if not current_members:
                current_members = [b]
                cur_min, cur_max = b.wl_min, b.wl_max
            else:
                gap = b.wl_min - (cur_max if cur_max is not None else b.wl_min)
                if gap <= max_gap:
                    # Merge
                    current_members.append(b)
                    cur_min = min(cur_min, b.wl_min) if cur_min is not None else b.wl_min
                    cur_max = max(cur_max, b.wl_max) if cur_max is not None else b.wl_max
                else:
                    clusters.append(
                        Cluster(molecule=mol, wl_min=cur_min, wl_max=cur_max, members=current_members[:])
                    )
                    current_members = [b]
                    cur_min, cur_max = b.wl_min, b.wl_max

        if current_members:
            clusters.append(
                Cluster(molecule=mol, wl_min=cur_min, wl_max=cur_max, members=current_members[:])
            )

    return clusters


def overlay_clusters_plot(
    wl: np.ndarray,
    T: np.ndarray,
    clusters: Sequence[Cluster],
    out_path: str,
    palette: Dict[str, str] | None = None,
) -> None:
    """
    Plot the transmission spectrum with semi-transparent rectangles denoting
    clustered absorption regions per molecule, using distinct colors.

    Saves a PNG to out_path.
    """
    if palette is None:
        # distinct hex colors for typical molecules
        palette = {
            "H2O": "#1f77b4",  # blue
            "CO2": "#d62728",  # red
            "CH4": "#2ca02c",  # green
            "CO": "#9467bd",   # purple
            "NH3": "#8c564b",  # brown
            "Na": "#e377c2",   # pink
            "K": "#7f7f7f",    # gray
        }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)

    # Line plot of the spectrum
    ax.plot(wl, T, color="black", lw=1.25, label="Transmission (simulated)")

    # Overlay clusters as translucent bands
    used_labels: set[Tuple[str, str]] = set()
    for c in clusters:
        color = palette.get(c.molecule, "#17becf")
        label = f"{c.molecule} clusters"
        # Deduplicate legend labels per molecule
        legend_label = label if (c.molecule, "legend") not in used_labels else None
        ax.axvspan(c.wl_min, c.wl_max, color=color, alpha=0.18, label=legend_label)
        used_labels.add((c.molecule, "legend"))

    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Transmission (relative)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(ncols=2, fontsize=9, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------ The Test -------------------------------------


@pytest.mark.mpl_image_compare(tolerance=50)  # optional if pytest-mpl is used; harmless otherwise
def test_absorption_overlay_clustered(tmp_path) -> str:
    """
    End-to-end diagnostic:
      1) Define canonical absorption bands for several molecules
      2) Simulate a transmission spectrum with substructure + noise
      3) Cluster nearby bands per molecule
      4) Render overlay plot, save artifact
      5) Assert scientific & rendering invariants

    Returns the path to the generated figure (interop with pytest-mpl).
    """
    seed_everything(123)

    # 1) Define a small library of band windows (μm). Values are illustrative
    #    (roughly near NIR regions used in exoplanet spectroscopy).
    bands = [
        Band("H2O", 1.35, 1.45),
        Band("H2O", 1.80, 1.95),
        Band("H2O", 2.60, 2.80),
        Band("CO2", 1.98, 2.10),
        Band("CO2", 4.20, 4.35),
        Band("CH4", 1.65, 1.73),
        Band("CH4", 2.30, 2.38),
        Band("CO",  2.29, 2.33),
        Band("NH3", 2.00, 2.08),
        Band("Na",  0.58, 0.60),
        Band("K",   0.76, 0.78),
    ]

    # 2) Simulate wavelength grid and transmission
    wl = np.linspace(0.5, 5.0, 1800)  # 0.5–5.0 μm
    T = simulate_transmission_spectrum(
        wl,
        bands=bands,
        line_depth=0.015,
        continuum_level=1.0,
        noise_ppm=60.0,
    )

    # 3) Cluster bands per molecule (merge gaps <= 0.08 μm)
    clusters = cluster_bands_by_proximity(bands, max_gap=0.08)

    # Basic cluster sanity: H2O has 3 windows that should remain 3 clusters (gaps > 0.08)
    n_h2o = sum(1 for c in clusters if c.molecule == "H2O")
    assert n_h2o == 3, f"Expected 3 H2O clusters, found {n_h2o}"

    # CH4 has bands at 1.65–1.73 and 2.30–2.38 (far apart) -> 2 clusters
    n_ch4 = sum(1 for c in clusters if c.molecule == "CH4")
    assert n_ch4 == 2, f"Expected 2 CH4 clusters, found {n_ch4}"

    # CO (2.29–2.33) and CH4 (2.30–2.38) overlap in wavelength region but are
    # different molecules => must remain separate clusters
    # Verify at least one CO cluster AND keep molecule separation.
    n_co = sum(1 for c in clusters if c.molecule == "CO")
    assert n_co == 1, "Expected 1 CO cluster"

    # 4) Render overlay
    out_png = os.path.join(tmp_path, "spectral_absorption_overlay_clustered.png")
    overlay_clusters_plot(wl, T, clusters, out_png)

    # 5) Scientific & rendering invariants

    # Transmission bounds in (0,1]
    assert float(np.min(T)) >= 0.0 - 1e-9, "Transmission has negative values"
    assert float(np.max(T)) <= 1.0 + 1e-9, "Transmission exceeds 1.0"

    # There should be visible dips around band centers (statistical check):
    for b in bands:
        mask = (wl >= b.wl_min) & (wl <= b.wl_max)
        band_median = float(np.median(T[mask]))
        cont_mask = (wl >= b.wl_min - 0.1) & (wl <= b.wl_min - 0.02)
        if np.sum(cont_mask) == 0:
            continue
        cont_median = float(np.median(T[cont_mask]))
        # Expect band median below nearby continuum (within tolerance)
        assert band_median < cont_median + 5e-4, (
            f"Band at {b.center:.2f} μm does not dip below continuum as expected"
        )

    # Ensure figure was written and is non-trivial
    assert os.path.isfile(out_png), "Output PNG not created"
    assert os.path.getsize(out_png) > 10_000, "Output PNG seems too small / empty"

    # Return path for pytest-mpl (if installed). If not using pytest-mpl,
    # the returned string is ignored by default pytest runner.
    return out_png


if __name__ == "__main__":
    # Allow quick manual run: python test_spectral_absorption_overlay_clustered.py
    # Will execute the test once and write artifact under ./tests_artifacts/
    seed_everything(123)
    tmp_dir = os.path.abspath("./tests_artifacts")
    os.makedirs(tmp_dir, exist_ok=True)
    # Run the test body
    _ = test_absorption_overlay_clustered(tmp_dir)
    print(f"[OK] Wrote overlay figure to: {os.path.join(tmp_dir, 'spectral_absorption_overlay_clustered.png')}")