"""
export_perpoint_breakouts.py
============================
Generate per-sample CSVs of caliper, baseline, threshold, excess from
threshold, and per-sample severity classification — the same arrays used
internally to render the per-sample severity bands in
`priority_wells_cumulative_min_v2_panel.png`.

This serializes the `perpoint` dictionary returned by
`detect_breakouts_cumulative_min` (which the panel uses but the previous
zone-level CSV did not include).

Output:
    /home/claude/work/outputs/priority_wells_cumulative_min_v2_perpoint.csv

Columns:
    well, depth_m, caliper_cm, baseline_cm, threshold_cm,
    excess_from_threshold_cm, severity_per_sample, zone_label
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cumulative_min_baseline import (
    fit_cumulative_min_split,
    detect_breakouts_cumulative_min,
)


# =============================================================================
#  CONFIG — identical to priority_wells_cumulative_min_v2.py
# =============================================================================

MASTER_CSV = Path("/home/claude/work/concatenate_caliper_all.csv")
NOISE_JSON = Path("/home/claude/work/outputs/noise_comparison.json")
OUT_DIR = Path("/home/claude/work/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIORITY_WELLS = ["AW5D", "AW6D", "BW3D", "LRS69D", "LRS70D"]

TRIM_DEPTHS_M = {
    "AW5D":   -5.0,
    "AW6D":   -5.0,
    "BW3D":   -7.0,
    "LRS69D": -7.0,
    "LRS70D": -5.0,
}

OFFSET_CM = 1.6
K_SIGMA = 1.0
L_MIN_M = 0.06
SATURATION_CM = 32.50

MILD_MAX_EXCESS_CM = 2.0 * OFFSET_CM        # 3.2 cm
MODERATE_MAX_EXCESS_CM = 6.0 * OFFSET_CM    # 9.6 cm


# =============================================================================
#  LOAD NOISE
# =============================================================================

with open(NOISE_JSON) as f:
    noise_report = json.load(f)
SIGMA_INST_CM = noise_report["AW5O"]["sigma_MAD_cm"]


# =============================================================================
#  LOAD MASTER CALIPER
# =============================================================================

df_master = pd.read_csv(MASTER_CSV)
df_master["well"] = df_master["source_file"].str.split("_").str[0]


# =============================================================================
#  PROCESS EACH WELL — capture the perpoint array
# =============================================================================

all_rows = []

for w in PRIORITY_WELLS:
    sub = df_master[df_master["well"] == w].copy()
    sub = sub.sort_values("Depth [m]").reset_index(drop=True)
    z = sub["Depth [m]"].to_numpy()
    cal = sub["calibrated_cm"].to_numpy()
    auger_in = float(sub["Diameter_auger_in"].iloc[0])
    auger_cm = auger_in * 2.54

    fit = fit_cumulative_min_split(
        z, cal,
        trim_depth_m=TRIM_DEPTHS_M[w],
        interp_kind="linear",
        direction="top_down",
        analyse_shallow=True,
        floor_cm=auger_cm,
        iqr_k=1.5,
    )
    zones, perpoint = detect_breakouts_cumulative_min(
        z, cal, fit.baseline,
        offset_cm=OFFSET_CM,
        sigma_inst_cm=SIGMA_INST_CM, k_sigma=K_SIGMA,
        L_min_m=L_MIN_M,
        saturation_cm=SATURATION_CM,
        mild_max_excess_cm=MILD_MAX_EXCESS_CM,
        moderate_max_excess_cm=MODERATE_MAX_EXCESS_CM,
        nominal_cm=auger_cm,
        zone_label=fit.zone_label,
    )

    # Build per-sample rows
    threshold_curve = perpoint["threshold_curve"]
    severity = perpoint["severity"]

    # Excess from threshold = caliper - threshold. Note this differs from
    # the array `perpoint["excess_over_thr"]` which is excess from
    # (B + offset) (without the noise term). The user explicitly asked for
    # excess from the FULL threshold, so we recompute.
    excess_from_threshold = cal - threshold_curve

    df_well = pd.DataFrame({
        "well": w,
        "depth_m": z,
        "caliper_cm": cal,
        "baseline_cm": fit.baseline,
        "threshold_cm": threshold_curve,
        "excess_from_threshold_cm": excess_from_threshold,
        "severity_per_sample": severity,
        "zone_label": fit.zone_label,
    })
    all_rows.append(df_well)

    # Quick per-well summary
    n_total = len(df_well)
    n_severe = int((severity == "severe").sum())
    n_mod = int((severity == "moderate").sum())
    n_mild = int((severity == "mild").sum())
    n_none = int((severity == "none").sum())
    print(f"{w:<8} n={n_total:>4d}  none={n_none:>4d}  "
          f"mild={n_mild:>4d}  moderate={n_mod:>3d}  severe={n_severe:>3d}")


# =============================================================================
#  CONCATENATE AND EXPORT
# =============================================================================

df_all = pd.concat(all_rows, ignore_index=True)

# Final column order, exactly as requested
cols = ["well", "depth_m", "caliper_cm", "baseline_cm", "threshold_cm",
        "excess_from_threshold_cm", "severity_per_sample", "zone_label"]
df_all = df_all[cols]

OUT_CSV = OUT_DIR / "priority_wells_cumulative_min_v2_perpoint.csv"
df_all.to_csv(OUT_CSV, index=False, float_format="%.4f")

print()
print(f"Total rows: {len(df_all)}")
print(f"Saved: {OUT_CSV}")

# Sanity check on LRS70D's severe-band heterogeneity (the case that motivated
# the per-sample export). Print a tight zoom around the deepest karst.
lrs70 = df_all[df_all["well"] == "LRS70D"]
karst = lrs70[(lrs70["depth_m"] > -22) & (lrs70["depth_m"] < -14.5)]
print()
print(f"Sanity check — LRS70D, depth range [-22, -14.5] m:")
print(f"  n samples       = {len(karst)}")
print(f"  none            = {(karst['severity_per_sample']=='none').sum()}")
print(f"  mild            = {(karst['severity_per_sample']=='mild').sum()}")
print(f"  moderate        = {(karst['severity_per_sample']=='moderate').sum()}")
print(f"  severe          = {(karst['severity_per_sample']=='severe').sum()}")
print(f"  This shows the heterogeneity inside the interval that previously")
print(f"  appeared as one monolithic 11-m severe block in the zone-level CSV.")
