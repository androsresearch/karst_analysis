"""Throwaway diagnostic for LRS70D feb-2022.

Purpose
-------
Investigate whether the input to the breakpoint detector is balanced in
depth, and how the raw YSI cast spacing translates into the resampled
uniform grid that the SavGol/LOWESS pipelines feed into
``best_n_breakpoints``.

Outputs
-------
results/figures/diagnostics/
    LRS70D_diag1_balance.png       — 4 panels: dz hist, depth-vs-index,
                                     resampled-grid hist, SEC + log10(SEC)
    LRS70D_diag2_density_overlay.png — smoothed log10(SEC) vs depth with
                                       grid ticks, SavGol method.
    LRS70D_diag3_raw_vs_resample.png — raw vs resampled depth-density
                                       comparison (kde + tick rug).

Run from the repo root:
    uv run python quick_diagnose_lrs70d.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.preprocessing import process_savgol


# ─── Inputs (hard-coded; this is throwaway) ────────────────────────────
RAW_CSV = Path("data/raw/sec/2022_02/LRS70_D_YSI_20220131.csv")
OUT_DIR = Path("results/figures/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WELL_LABEL = "LRS70D · 2022-01-31"


# ─── Load raw ──────────────────────────────────────────────────────────
df_raw = load_ysi_csv(RAW_CSV, standardise=True)
z_raw = df_raw["depth_m"].to_numpy()
sec_raw = df_raw["sec_uS_cm"].to_numpy()

# Sort raw by depth for dz statistics (the cast itself may not be sorted)
order = np.argsort(z_raw)
z_raw_sorted = z_raw[order]
dz_raw = np.diff(z_raw_sorted)
dz_p95 = np.percentile(dz_raw, 95)
dz_med = np.median(dz_raw)

# ─── Run SavGol pipeline (vadose=None: skips depth_bgl_m, irrelevant here) ─
df_sav, stats_sav = process_savgol(df_raw, vadose_thickness_m=None)
z_grid = df_sav["depth_m"].to_numpy()
log_sav = df_sav["log10_sec_uS_cm"].to_numpy()
sec_sav = df_sav["sec_uS_cm"].to_numpy()
dz_grid_actual = np.median(np.diff(z_grid))


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — 4-panel balance diagnostic
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"Balance diagnostic — {WELL_LABEL}\n"
    f"raw points: {len(df_raw)}   |   resampled grid: {len(df_sav)} pts at dz≈{dz_grid_actual:.4f} m",
    fontsize=12,
)

# Panel 1: dz raw histogram
ax = axes[0, 0]
ax.hist(dz_raw, bins=60, color="steelblue", edgecolor="white")
ax.axvline(dz_p95, color="red", lw=1.5, ls="--",
           label=f"p95 = {dz_p95:.4f} m  ← used as resample dz")
ax.axvline(dz_med, color="orange", lw=1.5, ls=":",
           label=f"median = {dz_med:.4f} m")
ax.set_xlabel("Δz between consecutive raw points (m)")
ax.set_ylabel("count")
ax.set_title("Panel 1 — Raw Δz distribution\n(diagnoses 'staircase' pauses)")
ax.legend(fontsize=9)

# Panel 2: depth vs sample index (cast structure in time)
ax = axes[0, 1]
ax.plot(np.arange(len(z_raw)), z_raw, ".", ms=2, color="steelblue", alpha=0.7)
ax.invert_yaxis()
ax.set_xlabel("Sample index (cast order)")
ax.set_ylabel("Depth (m)")
ax.set_title("Panel 2 — Depth vs sample index\n(visualises descent + stops)")
ax.grid(alpha=0.3)

# Panel 3: histogram of resampled-grid depths
ax = axes[1, 0]
ax.hist(z_grid, bins=60, color="seagreen", edgecolor="white")
ax.set_xlabel("Depth (m) — resampled grid that enters breakpoints")
ax.set_ylabel("count")
ax.set_title(f"Panel 3 — Resampled-grid depth histogram\n"
             f"(should be ~flat by construction; n={len(z_grid)})")

# Panel 4: SEC raw + log10(SEC) raw side-by-side histograms
ax = axes[1, 1]
ax.hist(sec_raw, bins=60, color="darkorange", alpha=0.6, label="SEC raw (linear)")
ax.set_xlabel("SEC (µS/cm)")
ax.set_ylabel("count (linear)", color="darkorange")
ax.tick_params(axis="y", labelcolor="darkorange")
ax2 = ax.twinx()
ax2.hist(np.log10(sec_raw[sec_raw > 0]), bins=60,
         color="navy", alpha=0.4, label="log10(SEC)")
ax2.set_ylabel("count (log10)", color="navy")
ax2.tick_params(axis="y", labelcolor="navy")
ax.set_title("Panel 4 — SEC raw value distribution\n"
             "(linear vs log10; diagnoses information density)")
ax.legend(loc="upper left", fontsize=9)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
out1 = OUT_DIR / "LRS70D_diag1_balance.png"
plt.savefig(out1, dpi=140, bbox_inches="tight")
plt.close()
print(f"  ✓ {out1}")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — smoothed log10(SEC) vs depth with grid ticks (SavGol)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 11))
ax.plot(log_sav, z_grid, "-", color="steelblue", lw=1.0,
        label="SavGol-smoothed log10(SEC)")
# Tick marks for each grid point (just below the trace)
tick_x = np.full_like(z_grid, log_sav.min() - 0.05, dtype=float)
ax.plot(tick_x, z_grid, "|", color="black", ms=4, mew=0.6, alpha=0.6,
        label=f"resample grid ({len(z_grid)} pts)")

ax.invert_yaxis()
ax.set_xlabel("log10(SEC) [µS/cm]")
ax.set_ylabel("Depth (m)")
ax.set_title(f"SavGol-smoothed log10(SEC) vs depth — {WELL_LABEL}\n"
             f"with input-grid density (ticks on left)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
out2 = OUT_DIR / "LRS70D_diag2_density_overlay.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
plt.close()
print(f"  ✓ {out2}")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — raw depth density vs resampled depth density
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(f"Depth density: raw vs resampled — {WELL_LABEL}",
             fontsize=12)

# Top: raw histogram + rug
ax = axes[0]
ax.hist(z_raw_sorted, bins=80, color="steelblue", alpha=0.6,
        edgecolor="white", label=f"raw histogram (n={len(z_raw_sorted)})")
ax.plot(z_raw_sorted,
        np.full_like(z_raw_sorted, -0.5, dtype=float),
        "|", color="black", ms=6, mew=0.4, alpha=0.5,
        label="raw point ticks")
ax.set_ylabel("count")
ax.set_title("Raw cast — bunched where the sonde paused")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

# Bottom: resampled histogram + rug
ax = axes[1]
ax.hist(z_grid, bins=80, color="seagreen", alpha=0.6,
        edgecolor="white", label=f"resampled histogram (n={len(z_grid)})")
ax.plot(z_grid,
        np.full_like(z_grid, -0.5, dtype=float),
        "|", color="black", ms=6, mew=0.4, alpha=0.5,
        label="grid ticks")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("count")
ax.set_title(f"Resampled grid — uniform by construction (dz≈{dz_grid_actual:.4f} m)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
out3 = OUT_DIR / "LRS70D_diag3_raw_vs_resample.png"
plt.savefig(out3, dpi=140, bbox_inches="tight")
plt.close()
print(f"  ✓ {out3}")


# ─── Console summary (for sanity) ──────────────────────────────────────
print()
print(f"Raw cast:       {len(df_raw)} points, depth {z_raw.min():.2f} → {z_raw.max():.2f} m")
print(f"Δz raw:         min={dz_raw.min():.4f}  median={dz_med:.4f}  "
      f"p95={dz_p95:.4f}  max={dz_raw.max():.4f}  (m)")
print(f"Resampled grid: {len(z_grid)} points at dz≈{dz_grid_actual:.4f} m")
print(f"SEC range:      {sec_raw.min():.0f} → {sec_raw.max():.0f} µS/cm")
print(f"log10(SEC):     {np.log10(sec_raw[sec_raw>0]).min():.2f} → "
      f"{np.log10(sec_raw[sec_raw>0]).max():.2f}")
