"""Side-by-side comparison of two smoothing back-ends.

Used to inform the per-well decision of which smoothing method best
preserves the features of interest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_smoothing_comparison(
    z_raw: np.ndarray,
    EC_raw: np.ndarray,
    z_savgol: np.ndarray,
    EC_savgol: np.ndarray,
    z_lowess: np.ndarray,
    EC_lowess: np.ndarray,
    *,
    output_path: str | Path,
    title: str = "Smoothing comparison",
    savgol_label: str = "SavGol",
    lowess_label: str = "LOWESS",
    figure_size: tuple = (10, 11),
    figure_dpi: int = 150,
    depth_axis_label: str = "Depth below ground level (m)",
    invert_y: bool = True,
    zoom_depth_range: Optional[tuple] = None,
) -> Path:
    """Two-panel figure: left = both methods overlaid on raw, right = optional zoom.

    The point of this figure is to make the visual decision easy: which
    smoother better preserves the transitions you care about?
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_panels = 2 if zoom_depth_range else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=figure_size, squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_panels},
    )
    ax_full = axes[0, 0]
    ax_zoom = axes[0, 1] if zoom_depth_range else None

    # Full panel.
    ax_full.scatter(EC_raw, z_raw, s=6, color="#bdc3c7", alpha=0.5,
                    linewidth=0, zorder=1, label=f"Raw ({len(z_raw):,})")
    ax_full.plot(EC_savgol, z_savgol, color="#c0392b", lw=1.4, zorder=3,
                 label=f"{savgol_label} ({len(z_savgol):,})")
    ax_full.plot(EC_lowess, z_lowess, color="#1f4e79", lw=1.4, zorder=4,
                 label=f"{lowess_label} ({len(z_lowess):,})")
    ax_full.set_xlabel("Specific electrical conductivity")
    ax_full.set_ylabel(depth_axis_label)
    ax_full.grid(True, ls=":", alpha=0.5)
    ax_full.legend(loc="best", fontsize=9, framealpha=0.92)
    ax_full.set_title("Full profile")
    if invert_y:
        ax_full.invert_yaxis()

    if zoom_depth_range:
        z_min, z_max = sorted(zoom_depth_range)
        ax_zoom.scatter(EC_raw, z_raw, s=10, color="#bdc3c7", alpha=0.5,
                        linewidth=0, zorder=1, label="Raw")
        ax_zoom.plot(EC_savgol, z_savgol, color="#c0392b", lw=1.6, zorder=3,
                     label=savgol_label)
        ax_zoom.plot(EC_lowess, z_lowess, color="#1f4e79", lw=1.6, zorder=4,
                     label=lowess_label)
        ax_zoom.set_xlabel("Specific electrical conductivity")
        ax_zoom.set_ylabel(depth_axis_label)
        ax_zoom.grid(True, ls=":", alpha=0.5)
        ax_zoom.set_title(f"Zoom — z ∈ [{z_min:.2f}, {z_max:.2f}] m")
        ax_zoom.set_ylim(z_min, z_max)
        if invert_y:
            ax_zoom.invert_yaxis()
        ax_zoom.legend(loc="best", fontsize=9, framealpha=0.92)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out
