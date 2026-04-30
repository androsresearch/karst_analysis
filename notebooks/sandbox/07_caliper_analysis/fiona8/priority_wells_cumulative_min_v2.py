"""
Breakouts detector configuration:

    * Baseline:    cumulative-minimum (top-down, linear interpolation),
                   shallow/deep split with per-well trim depth.
    * Threshold:   thr(z) = baseline(z) + OFFSET_CM + K_SIGMA * sigma_inst
                   (fixed additive offset in cm calculated over AW5O)
    * Severity:    "severe"   = caliper saturated inside zone
                   "moderate" = peak_excess_cm >= MODERATE_EXCESS_CM
                   "mild"     = otherwise (just above threshold)

The fixed offset of 1.6 cm makes the new rule recover removes the dependency on
well diameter and gives a transparent physical meaning to the threshold:
"a breakout is when the hole is at least OFFSET_CM cm wider than the local
baseline, sustained over at least L_min metres".

Run:
    uv run python .\notebooks\sandbox\07_caliper_analysis\fiona8\priority_wells_cumulative_min_v2.py

Inputs 
    - concatenate_caliper_all.csv   master caliper file
    - outputs/noise_comparison.json sigma_inst from AW5O
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from cumulative_min_baseline import (
    fit_cumulative_min_split,
    detect_breakouts_cumulative_min,
)


# =============================================================================
#  CONFIG
# =============================================================================

MASTER_CSV = Path(r"data\caliper\concatenate_caliper_all.csv")
NOISE_JSON = Path(r"notebooks\sandbox\07_caliper_analysis\fiona8\outputs\noise_comparison.json")
OUT_DIR = Path(r"notebooks\sandbox\07_caliper_analysis\fiona8\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "priority_wells_cumulative_min_v2_panel.png"
OUT_CSV = OUT_DIR / "priority_wells_cumulative_min_v2_zones.csv"

PRIORITY_WELLS = ["AW5D", "AW6D", "BW3D", "LRS69D", "LRS70D"]

# Per-well trim depths
TRIM_DEPTHS_M = {
    "AW5D":   -5.0,
    "AW6D":   -5.0,
    "BW3D":   -7.0,
    "LRS69D": -7.0,
    "LRS70D": -5.0,
}

# Detection rule:  C(z) > B(z) + OFFSET_CM + K_SIGMA * sigma_inst
OFFSET_CM = 1.6
K_SIGMA = 1.0
L_MIN_M = 0.06
SATURATION_CM = 32.50

# Severity classification by peak_excess_cm (= peak_cm - baseline_at_peak):
#   severe   : saturated  OR  peak_excess >= MODERATE_MAX_EXCESS_CM
#   moderate : MILD_MAX_EXCESS_CM <= peak_excess < MODERATE_MAX_EXCESS_CM
#   mild     : peak_excess < MILD_MAX_EXCESS_CM
MILD_MAX_EXCESS_CM = 2.0 * OFFSET_CM       # 3.2 cm
MODERATE_MAX_EXCESS_CM = 6.0 * OFFSET_CM   # 9.6 cm


# =============================================================================
#  LOAD NOISE
# =============================================================================

with open(NOISE_JSON) as f:
    noise_report = json.load(f)
SIGMA_INST_CM = noise_report["AW5O"]["sigma_MAD_cm"]
NOISE_INTERVAL = noise_report["AW5O"]["well_interval"]

print("=" * 78)
print("PRIORITY-WELL ANALYSIS — cumulative-min baseline + fixed-cm offset")
print("=" * 78)
print(f"\nWells (left to right):  {' -> '.join(PRIORITY_WELLS)}")
print(f"\nNoise reference: AW5O sigma_MAD over {NOISE_INTERVAL} m = "
      f"{SIGMA_INST_CM:.4f} cm")
print(f"Detection rule:  C(z) > B(z) + {OFFSET_CM:.1f} cm "
      f"+ {K_SIGMA:.0f} * {SIGMA_INST_CM:.4f}")
print(f"Per-sample severity (excess measured FROM THRESHOLD = B + offset):")
print(f"  severe   : excess_over_thr >= {MODERATE_MAX_EXCESS_CM:.2f} cm OR saturated")
print(f"  moderate : {MILD_MAX_EXCESS_CM:.2f} <= excess_over_thr < "
      f"{MODERATE_MAX_EXCESS_CM:.2f} cm")
print(f"  mild     : 0 < excess_over_thr < {MILD_MAX_EXCESS_CM:.2f} cm")
print(f"L_min = {L_MIN_M:.2f} m")


# =============================================================================
#  LOAD WELLS
# =============================================================================

df_master = pd.read_csv(MASTER_CSV)
df_master["well"] = df_master["source_file"].str.split("_").str[0]

well_data = {}
for w in PRIORITY_WELLS:
    sub = df_master[df_master["well"] == w].copy()
    sub = sub.sort_values("Depth [m]").reset_index(drop=True)
    z = sub["Depth [m]"].to_numpy()
    cal = sub["calibrated_cm"].to_numpy()
    auger_in = float(sub["Diameter_auger_in"].iloc[0])
    auger_cm = auger_in * 2.54
    well_data[w] = dict(z=z, cal=cal, auger_in=auger_in, auger_cm=auger_cm,
                          n=len(z))


# =============================================================================
#  FIT + DETECT
# =============================================================================

results = {}
for w in PRIORITY_WELLS:
    d = well_data[w]
    fit = fit_cumulative_min_split(
        d["z"], d["cal"],
        trim_depth_m=TRIM_DEPTHS_M[w],
        interp_kind="linear",
        direction="top_down",
        analyse_shallow=True,
        floor_cm=d["auger_cm"],
        iqr_k=1.5,
    )
    zones, perpoint = detect_breakouts_cumulative_min(
        d["z"], d["cal"], fit.baseline,
        offset_cm=OFFSET_CM,
        sigma_inst_cm=SIGMA_INST_CM, k_sigma=K_SIGMA,
        L_min_m=L_MIN_M,
        saturation_cm=SATURATION_CM,
        mild_max_excess_cm=MILD_MAX_EXCESS_CM,
        moderate_max_excess_cm=MODERATE_MAX_EXCESS_CM,
        nominal_cm=d["auger_cm"],
        zone_label=fit.zone_label,
    )
    n_below_auger = int((d["cal"] < d["auger_cm"]).sum())
    results[w] = dict(fit=fit, zones=zones, perpoint=perpoint,
                      trim=TRIM_DEPTHS_M[w],
                      n_below_auger=n_below_auger)


# Summary
print(f"\n{'='*78}")
print("DETECTION SUMMARY")
print(f"{'='*78}")
print(f"\n{'Well':<8} {'Auger':>7} {'n':>5} {'<auger':>7} "
      f"{'severe':>8} {'moderate':>10} {'mild':>6} {'total':>7}")
print("-" * 78)
for w in PRIORITY_WELLS:
    d = well_data[w]
    r = results[w]
    zones = r["zones"]
    n_severe = sum(1 for zn in zones if zn["severity"] == "severe")
    n_mod = sum(1 for zn in zones if zn["severity"] == "moderate")
    n_mild = sum(1 for zn in zones if zn["severity"] == "mild")
    print(f"{w:<8} {d['auger_in']:>5.0f}\"  {d['n']:>5d} "
          f"{r['n_below_auger']:>7d} "
          f"{n_severe:>8d} {n_mod:>10d} {n_mild:>6d} {len(zones):>7d}")
print()


# =============================================================================
#  PANEL FIGURE
# =============================================================================

# Severity colour scheme (traffic light)
COLOR_SEVERE = "#c0392b"     # red
COLOR_MODERATE = "#f39c12"   # yellow / amber
COLOR_MILD = "#fde3a7"       # light orange / very pale

# Common axis ranges
x_lo = max(0, min(d["cal"].min() for d in well_data.values()) - 1)
x_hi = max(SATURATION_CM,
           max(d["cal"].max() for d in well_data.values())) + 1.5
y_lo = min(d["z"].min() for d in well_data.values()) - 1
y_hi = max(d["z"].max() for d in well_data.values()) + 0.5

n_wells = len(PRIORITY_WELLS)
fig, axes = plt.subplots(1, n_wells, figsize=(4.2 * n_wells, 18),
                          sharey=True,
                          gridspec_kw={"wspace": 0.06})

MIN_VISUAL_HEIGHT = 0.10

for ax, w in zip(axes, PRIORITY_WELLS):
    d = well_data[w]
    fit = results[w]["fit"]
    zones = results[w]["zones"]
    z = d["z"]; cal = d["cal"]
    auger_cm = d["auger_cm"]
    trim_w = results[w]["trim"]

    # Validity filter (same rule used by the fitter), for plotting:
    # break the caliper trace at invalid samples.
    valid = cal >= auger_cm
    if np.any(valid):
        q1 = np.nanpercentile(cal[valid], 25)
        q3 = np.nanpercentile(cal[valid], 75)
        iqr = q3 - q1
        valid &= cal >= (q1 - 1.5 * iqr)
    n_dropped = int((~valid).sum())

    order = np.argsort(z)
    z_s, cal_s = z[order], cal[order]
    base_s = fit.baseline[order]
    label_s = fit.zone_label[order]
    valid_s = valid[order]

    # Shallow zone tinted background
    is_shallow_sample = label_s == "shallow"
    if is_shallow_sample.any():
        z_sh = z_s[is_shallow_sample]
        ax.axhspan(z_sh.min(), z_sh.max(),
                   color="#fff7e6", alpha=0.55, zorder=0)

    # Caliper trace, breaking line at invalid samples
    cal_plot = np.where(valid_s, cal_s, np.nan)
    ax.plot(cal_plot, z_s, color="#8e6914", lw=0.6, alpha=0.85, zorder=2)

    # Saturated points (on valid samples)
    sat_mask = valid_s & (cal_s >= SATURATION_CM)
    if sat_mask.any():
        ax.scatter(cal_s[sat_mask], z_s[sat_mask],
                   s=4, color="#c0392b", alpha=0.7, zorder=3)

    # Baseline and threshold drawn separately per sub-zone (no fake link
    # across the trim depth)
    for zone_name in ("shallow", "deep"):
        mask = label_s == zone_name
        if not mask.any():
            continue
        zz = z_s[mask]; bb = base_s[mask]
        tt = bb + OFFSET_CM + K_SIGMA * SIGMA_INST_CM
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
    ax.axvline(auger_cm, color="#888888", ls=":", lw=1.0, alpha=0.7, zorder=1)
    ax.axvline(SATURATION_CM, color="#8B0000", ls="-", lw=0.8,
               alpha=0.5, zorder=1)
    ax.axhline(trim_w, color="#666666", ls="-.", lw=0.7,
               alpha=0.6, zorder=1)

    # Per-sample severity bands (replaces the old monolithic zone blocks).
    # Each flagged sample gets a thin horizontal band of half-width dz/2,
    # coloured by its individual severity. This shows the internal
    # heterogeneity of long zones (e.g. LRS70D karst) instead of pretending
    # the whole zone is one solid colour.
    perpoint = results[w]["perpoint"]
    sev_per_pt = perpoint["severity"][order]   # aligned to z_s
    dz_local = float(np.median(np.diff(np.sort(z_s))))
    half_dz = dz_local / 2.0

    for sev_name, color, alpha in [
        ("mild", COLOR_MILD, 0.65),
        ("moderate", COLOR_MODERATE, 0.55),
        ("severe", COLOR_SEVERE, 0.55),
    ]:
        idx = np.flatnonzero(sev_per_pt == sev_name)
        for i in idx:
            ax.axhspan(z_s[i] - half_dz, z_s[i] + half_dz,
                       color=color, alpha=alpha, zorder=1.5,
                       linewidth=0)

    # Zone counts by severity (zone-level, for the title)
    n_severe = sum(1 for zn in zones if zn["severity"] == "severe")
    n_mod = sum(1 for zn in zones if zn["severity"] == "moderate")
    n_mild = sum(1 for zn in zones if zn["severity"] == "mild")

    # Title
    drop_note = f" [-{n_dropped} artefacts]" if n_dropped > 0 else ""
    ax.set_title(
        f"{w}\n"
        f"{d['auger_in']:.0f}\" auger ({auger_cm:.2f} cm)\n"
        f"{drop_note}\n"
        f"{len(zones)} zones — sev:{n_severe} mod:{n_mod} mild:{n_mild}",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Caliper [cm]", fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

axes[0].set_ylabel("Depth [m]", fontsize=13)

# Legend
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
    Line2D([0], [0], color="#888888", ls=":", lw=1.0, label="Auger nominal"),
    Line2D([0], [0], color="#8B0000", lw=1.0, label="Saturation 32.5 cm"),
    Line2D([0], [0], color="#666666", ls="-.", lw=1.0,
           label="Trim depth (per well)"),
    Patch(facecolor="#fff7e6", alpha=0.55, label="Shallow zone"),
    Patch(facecolor=COLOR_SEVERE, alpha=0.55,
          label=f"Severe (saturated or excess from thr ≥ {MODERATE_MAX_EXCESS_CM:.1f} cm)"),
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
    f"sigma = {SIGMA_INST_CM:.4f} cm (AW5O sigma_MAD, [-5, -1] m), "
    f"L_min = {L_MIN_M:.2f} m   |   "
    f"severity from THRESHOLD: mild < {MILD_MAX_EXCESS_CM:.1f} cm, "
    f"moderate < {MODERATE_MAX_EXCESS_CM:.1f} cm, severe / sat",
    fontsize=13, fontweight="bold", y=1.045,
)
fig.tight_layout(rect=[0, 0, 1, 0.86])

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: {OUT_PNG}")

# Export zones
rows = []
for w in PRIORITY_WELLS:
    for zn in results[w]["zones"]:
        rows.append({"well": w, **zn})
df_zones = pd.DataFrame(rows)
df_zones.to_csv(OUT_CSV, index=False)
print(f"✓ Saved: {OUT_CSV}")

plt.close("all")
print("\nDone.\n")
