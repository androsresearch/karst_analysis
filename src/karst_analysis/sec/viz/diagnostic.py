"""Diagnostic plots: raw-vs-processed comparison and sampling balance.

Adapted from the LOWESS preprocessing module. Generic enough to be used
for any smoothing back-end — the figure title and method-info textbox
adapt to the parameters passed in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostic(
    z_raw: np.ndarray,
    EC_raw: np.ndarray,
    z_proc: np.ndarray,
    EC_proc: np.ndarray,
    *,
    output_path: str | Path,
    title: str = "SEC pre-processing",
    method_info: str = "",
    show_zoom_panel: bool = False,
    zoom_depth_range: Optional[tuple] = None,
    breakpoints: Optional[list[tuple[float, float]]] = None,
    figure_size: tuple = (8, 11),
    figure_dpi: int = 150,
    depth_axis_label: Optional[str] = None,
    invert_y: bool = True,
    vadose_offset_m: float = 0.0,
) -> Path:
    """Plot raw vs processed profile, optionally with zoom and breakpoints.

    Parameters
    ----------
    z_raw, EC_raw : np.ndarray
        Raw profile (used as the grey scatter background). Depths in
        whatever datum the caller chose; pass ``vadose_offset_m`` to
        convert from water-table datum to BGL.
    z_proc, EC_proc : np.ndarray
        Processed profile (overlaid as a line).
    output_path : str or Path
        Where to save the PNG.
    title : str
        Figure title.
    method_info : str
        Multi-line string shown in the lower-left textbox; describes the
        pipeline that was applied.
    show_zoom_panel : bool, default False
        If True, append a second panel zoomed into ``zoom_depth_range``.
    zoom_depth_range : (z_min, z_max), required if ``show_zoom_panel``.
        Interpreted in the SAME datum as ``z_raw`` / ``z_proc`` AFTER
        the vadose offset is applied.
    breakpoints : list of (depth, EC) tuples, optional
        Markers overlaid on both panels. Depths are shifted by the same
        ``vadose_offset_m`` as the profiles.
    figure_size, figure_dpi : matplotlib output controls.
    depth_axis_label : str, optional
        Custom y-axis label. If ``None`` (default), derived from
        ``vadose_offset_m`` — see that parameter.
    invert_y : bool, default True
        If True, depth grows downward (typical for borehole plots).
    vadose_offset_m : float, default 0.0
        Vertical offset (in metres) added to every depth value before
        plotting, to convert the SEC pipeline's native water-table
        datum to below-ground-level. Pass the well's
        ``vadose_thickness_m`` from ``data/metadata/wells.csv``. When
        ``depth_axis_label`` is ``None``, the y-label is auto-derived:
        ``"Depth below ground level (m)"`` for nonzero offset, else
        ``"Depth below water table (m)"``. BGL is the canonical datum
        for karst_analysis (see CHANGELOG v17.3).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if show_zoom_panel and zoom_depth_range is None:
        raise ValueError("show_zoom_panel requires zoom_depth_range=(z_min, z_max)")

    # ── Datum shift ----------------------------------------------------
    z_raw = np.asarray(z_raw, dtype=float) + vadose_offset_m
    z_proc = np.asarray(z_proc, dtype=float) + vadose_offset_m
    if breakpoints is not None and vadose_offset_m != 0.0:
        breakpoints = [(d + vadose_offset_m, ec) for (d, ec) in breakpoints]
    if zoom_depth_range is not None and vadose_offset_m != 0.0:
        zoom_depth_range = (
            zoom_depth_range[0] + vadose_offset_m,
            zoom_depth_range[1] + vadose_offset_m,
        )
    if depth_axis_label is None:
        depth_axis_label = (
            "Depth below ground level (m)" if vadose_offset_m > 0
            else "Depth below water table (m)"
        )

    n_panels = 2 if show_zoom_panel else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=figure_size, squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_panels},
    )
    ax_full = axes[0, 0]
    ax_zoom = axes[0, 1] if show_zoom_panel else None

    # Full panel.
    ax_full.scatter(EC_raw, z_raw, s=6, color="#bdc3c7", alpha=0.55,
                    linewidth=0, zorder=1, label=f"Raw ({len(z_raw):,} pts)")
    ax_full.plot(EC_proc, z_proc, color="#1f4e79", lw=1.4, zorder=3,
                 label=f"Processed ({len(z_proc):,} pts)")
    if breakpoints:
        bp_arr = np.asarray(breakpoints, dtype=float)
        ax_full.scatter(bp_arr[:, 1], bp_arr[:, 0], s=70, marker="D",
                        facecolor="#e67e22", edgecolor="black", linewidth=0.7,
                        zorder=5, label=f"Breakpoints ({len(bp_arr)})")
    ax_full.set_xlabel("Specific electrical conductivity")
    ax_full.set_ylabel(depth_axis_label)
    ax_full.grid(True, ls=":", alpha=0.5)
    ax_full.legend(loc="best", fontsize=9, framealpha=0.92)
    ax_full.set_title("Full profile")
    if invert_y:
        ax_full.invert_yaxis()

    if method_info:
        ax_full.text(0.02, 0.02, method_info, transform=ax_full.transAxes,
                     fontsize=8, family="monospace", va="bottom", ha="left",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="#1f4e79", alpha=0.92))

    # Zoom panel.
    if show_zoom_panel:
        z_min, z_max = sorted(zoom_depth_range)

        ax_zoom.scatter(EC_raw, z_raw, s=10, color="#bdc3c7", alpha=0.55,
                        linewidth=0, zorder=1, label="Raw")
        ax_zoom.plot(EC_proc, z_proc, color="#1f4e79", lw=1.6, zorder=3,
                     label="Processed")
        if breakpoints:
            bp_arr = np.asarray(breakpoints, dtype=float)
            in_zoom = (bp_arr[:, 0] >= z_min) & (bp_arr[:, 0] <= z_max)
            if in_zoom.any():
                ax_zoom.scatter(
                    bp_arr[in_zoom, 1], bp_arr[in_zoom, 0],
                    s=80, marker="D", facecolor="#e67e22", edgecolor="black",
                    linewidth=0.7, zorder=5,
                    label=f"Breakpoints ({int(in_zoom.sum())})",
                )
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


def plot_balance_histogram(
    z_raw: np.ndarray,
    z_proc: np.ndarray,
    *,
    output_path: str | Path,
    n_bins: int = 40,
    figure_size: tuple = (10, 5),
    figure_dpi: int = 150,
    depth_axis_label: Optional[str] = None,
    vadose_offset_m: float = 0.0,
) -> Path:
    """Side-by-side histograms of point counts per depth bin (raw vs processed).

    Useful for verifying whether resampling improved sampling balance.

    Parameters
    ----------
    vadose_offset_m : float, default 0.0
        Vertical offset (in metres) added to depth values before
        binning, to convert water-table datum to BGL. See
        :func:`plot_diagnostic` for full semantics. BGL is the
        canonical datum for karst_analysis (see CHANGELOG v17.3).
    depth_axis_label : str, optional
        Custom y-axis label. If ``None``, derived from
        ``vadose_offset_m``.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    z_raw = np.asarray(z_raw, dtype=float) + vadose_offset_m
    z_proc = np.asarray(z_proc, dtype=float) + vadose_offset_m
    if depth_axis_label is None:
        depth_axis_label = (
            "Depth below ground level (m)" if vadose_offset_m > 0
            else "Depth below water table (m)"
        )

    z_min = float(min(z_raw.min(), z_proc.min()))
    z_max = float(max(z_raw.max(), z_proc.max()))
    bins = np.linspace(z_min, z_max, n_bins + 1)
    bin_w = (z_max - z_min) / n_bins

    h_raw, _ = np.histogram(z_raw, bins=bins)
    h_proc, _ = np.histogram(z_proc, bins=bins)

    def imbalance(h):
        nz = h[h > 0]
        return (nz.max() / np.median(nz)) if len(nz) else np.nan

    imb_raw = imbalance(h_raw)
    imb_proc = imbalance(h_proc)

    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharey=True)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for ax, h, color, name, imb in [
        (axes[0], h_raw, "#7f8c8d", "Raw", imb_raw),
        (axes[1], h_proc, "#1f4e79", "Processed", imb_proc),
    ]:
        ax.barh(centers, h, height=(bins[1] - bins[0]) * 0.95,
                color=color, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Points per bin")
        ax.set_title(f"{name}   (max/median = {imb:.1f})", fontsize=10)
        ax.grid(True, axis="x", ls=":", alpha=0.5)

    axes[0].set_ylabel(depth_axis_label)
    fig.suptitle(
        f"Sampling balance — bin width {bin_w:.2f} m ({n_bins} bins). "
        f"Lower max/median ⇒ more uniform.",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out
