"""Visualisation of the priority-wells caliper panel.

Reproduces ``priority_wells_cumulative_min_v2_panel.png`` of the
original pipeline, with one column per well showing:

    * caliper trace (with invalid samples broken out)
    * baseline + threshold (per sub-zone)
    * shallow / deep zone tinted background
    * per-sample severity bands (mild / moderate / severe)
    * anchor points for the cumulative-min fit

Migration history
-----------------
v5: extracted from ``priority_wells_cumulative_min_v2.py`` with no
algorithmic changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from karst_analysis.caliper.config import (
    OFFSET_CM, K_SIGMA, L_MIN_M, SATURATION_CM,
    MILD_MAX_EXCESS_CM, MODERATE_MAX_EXCESS_CM,
)


# Severity colour scheme (traffic light)
COLOR_SEVERE   = "#c0392b"   # red
COLOR_MODERATE = "#f39c12"   # amber
COLOR_MILD     = "#fde3a7"   # very pale orange


def plot_priority_wells_panel(
    results: dict[str, dict],
    sigma_inst_cm: float,
    *,
    output_path: str | Path,
    well_order: Optional[list[str]] = None,
    figure_height: float = 18.0,
    width_per_well: float = 4.2,
    dpi: int = 150,
) -> Path:
    """Render the multi-well panel figure.

    Parameters
    ----------
    results : dict
        Output of :func:`karst_analysis.caliper.pipeline.process_many_wells`.
    sigma_inst_cm : float
        Instrumental noise (cm). Shown in the figure title.
    output_path : str or Path
        Where the PNG goes.
    well_order : list of str, optional
        Order of wells from left to right. If None, uses ``results.keys()``.
    figure_height : float
        Inches of vertical extent.
    width_per_well : float
        Inches per well column.
    dpi : int
        Output resolution.

    Returns
    -------
    Path
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if well_order is None:
        well_order = list(results.keys())

    # Common axis ranges across panels
    x_lo = max(0, min(r["cal"].min() for r in results.values()) - 1)
    x_hi = max(SATURATION_CM,
               max(r["cal"].max() for r in results.values())) + 1.5
    # BGL-positive: y_lo is near the surface (~0), y_hi is the deepest point.
    # The axis is inverted afterward so 0 ends up at the top of the figure.
    y_lo = min(r["z"].min() for r in results.values()) - 0.5
    y_hi = max(r["z"].max() for r in results.values()) + 1.0

    n_wells = len(well_order)
    fig, axes = plt.subplots(
        1, n_wells, figsize=(width_per_well * n_wells, figure_height),
        sharey=True, gridspec_kw={"wspace": 0.06},
    )
    if n_wells == 1:
        axes = [axes]

    for ax, well_id in zip(axes, well_order):
        r = results[well_id]
        z = r["z"]
        cal = r["cal"]
        auger_cm = r["auger_cm"]
        trim_w = r["trim_depth_m"]
        fit = r["fit"]
        zones = r["zones"]

        # Validity filter (same as the fitter), for plotting the trace:
        # break the caliper line at invalid samples.
        valid = cal >= auger_cm
        if np.any(valid):
            q1 = np.nanpercentile(cal[valid], 25)
            q3 = np.nanpercentile(cal[valid], 75)
            iqr = q3 - q1
            valid &= cal >= (q1 - 1.5 * iqr)
        n_dropped = int((~valid).sum())

        order = np.argsort(z)
        z_s = z[order]
        cal_s = cal[order]
        base_s = fit.baseline[order]
        label_s = fit.zone_label[order]
        valid_s = valid[order]

        # Shallow zone tinted background
        is_shallow_sample = label_s == "shallow"
        if is_shallow_sample.any():
            z_sh = z_s[is_shallow_sample]
            ax.axhspan(z_sh.min(), z_sh.max(),
                       color="#fff7e6", alpha=0.55, zorder=0)

        # Caliper trace, breaking at invalid samples
        cal_plot = np.where(valid_s, cal_s, np.nan)
        ax.plot(cal_plot, z_s, color="#8e6914", lw=0.6, alpha=0.85, zorder=2)

        # Saturated points
        sat_mask = valid_s & (cal_s >= SATURATION_CM)
        if sat_mask.any():
            ax.scatter(cal_s[sat_mask], z_s[sat_mask],
                       s=4, color="#c0392b", alpha=0.7, zorder=3)

        # Baseline and threshold per sub-zone (no fake link across trim)
        for zone_name in ("shallow", "deep"):
            mask = label_s == zone_name
            if not mask.any():
                continue
            zz = z_s[mask]
            bb = base_s[mask]
            tt = bb + OFFSET_CM + K_SIGMA * sigma_inst_cm
            ax.plot(bb, zz, color="#1d4ed8", lw=1.4, zorder=4, alpha=0.95)
            ax.plot(tt, zz, color="#27ae60", lw=1.3, ls="--",
                    zorder=5, alpha=0.9)

        # Anchor points
        if fit.shallow is not None and fit.shallow.n_anchors > 0:
            ax.plot(fit.shallow.anchor_cal, fit.shallow.anchor_z,
                    "o", ms=4.5, mec="#1d4ed8", mfc="white",
                    mew=1.0, zorder=6)
        if fit.deep is not None and fit.deep.n_anchors > 0:
            ax.plot(fit.deep.anchor_cal, fit.deep.anchor_z,
                    "o", ms=4.5, mec="#1d4ed8", mfc="#1d4ed8", zorder=6)

        # Reference lines
        ax.axvline(auger_cm, color="#888888", ls=":", lw=1.0,
                   alpha=0.7, zorder=1)
        ax.axvline(SATURATION_CM, color="#8B0000", ls="-", lw=0.8,
                   alpha=0.5, zorder=1)
        ax.axhline(trim_w, color="#666666", ls="-.", lw=0.7,
                   alpha=0.6, zorder=1)

        # Per-sample severity bands
        sev_per_pt = r["perpoint"]["severity"][order]
        dz_local = float(np.median(np.diff(np.sort(z_s))))
        half_dz = dz_local / 2.0

        for sev_name, color, alpha in [
            ("mild",     COLOR_MILD,     0.65),
            ("moderate", COLOR_MODERATE, 0.55),
            ("severe",   COLOR_SEVERE,   0.55),
        ]:
            idx = np.flatnonzero(sev_per_pt == sev_name)
            for i in idx:
                ax.axhspan(z_s[i] - half_dz, z_s[i] + half_dz,
                           color=color, alpha=alpha, zorder=1.5,
                           linewidth=0)

        # Title with zone counts
        n_severe = sum(1 for zn in zones if zn["severity"] == "severe")
        n_mod = sum(1 for zn in zones if zn["severity"] == "moderate")
        n_mild = sum(1 for zn in zones if zn["severity"] == "mild")
        drop_note = f" [-{n_dropped} artefacts]" if n_dropped > 0 else ""
        ax.set_title(
            f"{well_id}\n"
            f"{r['auger_in']:.0f}\" auger ({auger_cm:.2f} cm)\n"
            f"{drop_note}\n"
            f"{len(zones)} zones — sev:{n_severe} mod:{n_mod} mild:{n_mild}",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("Caliper [cm]", fontsize=11)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Depth below ground level (m)", fontsize=13)
    # With BGL-positive convention, 0 should be at the top.
    # sharey=True means a single invert call flips both panels.
    axes[0].invert_yaxis()

    legend_elements = [
        Line2D([0], [0], color="#8e6914", lw=1.0, label="Caliper signal"),
        Line2D([0], [0], color="#1d4ed8", lw=1.5,
               label="Baseline M(z) — cumulative min, linear"),
        Line2D([0], [0], marker="o", mec="#1d4ed8", mfc="#1d4ed8",
               ls="None", ms=6, label="Deep anchor"),
        Line2D([0], [0], marker="o", mec="#1d4ed8", mfc="white", mew=1.2,
               ls="None", ms=6, label="Shallow anchor"),
        Line2D([0], [0], color="#27ae60", lw=1.5, ls="--",
               label=f"Threshold = B(z) + {OFFSET_CM:.1f} cm + {K_SIGMA:.0f}σ"),
        Line2D([0], [0], color="#888888", ls=":", lw=1.0,
               label="Auger nominal"),
        Line2D([0], [0], color="#8B0000", lw=1.0,
               label=f"Saturation {SATURATION_CM:.1f} cm"),
        Line2D([0], [0], color="#666666", ls="-.", lw=1.0,
               label="Trim depth (per well)"),
        Patch(facecolor="#fff7e6", alpha=0.55, label="Shallow zone"),
        Patch(facecolor=COLOR_SEVERE, alpha=0.55,
              label=f"Severe (sat or excess from thr ≥ {MODERATE_MAX_EXCESS_CM:.1f} cm)"),
        Patch(facecolor=COLOR_MODERATE, alpha=0.55,
              label=f"Moderate ({MILD_MAX_EXCESS_CM:.1f} ≤ excess from thr < {MODERATE_MAX_EXCESS_CM:.1f} cm)"),
        Patch(facecolor=COLOR_MILD, alpha=0.65,
              label=f"Mild (0 < excess from thr < {MILD_MAX_EXCESS_CM:.1f} cm)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.992),
               ncol=4, fontsize=11, framealpha=0.95)

    fig.suptitle(
        "Cumulative-minimum baseline (top-down, linear) — "
        "fixed-cm offset + per-sample severity bands\n"
        f"offset = {OFFSET_CM:.1f} cm, k = {K_SIGMA:.0f}, "
        f"sigma = {sigma_inst_cm:.4f} cm, L_min = {L_MIN_M:.2f} m   |   "
        f"severity from THRESHOLD: mild < {MILD_MAX_EXCESS_CM:.1f} cm, "
        f"moderate < {MODERATE_MAX_EXCESS_CM:.1f} cm, severe / sat",
        fontsize=13, fontweight="bold", y=1.045,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
