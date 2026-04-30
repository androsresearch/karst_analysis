"""
estimate_noise_aw5o_vs_aw5d.py — Comparable instrumental-noise estimates
on AW5O and AW5D, using the SAME procedure on different fixed intervals.

Rationale
---------
AW5O was drilled with sonic coring (smooth borehole wall, minimal
formation disturbance); AW5D was drilled with rotary auger (highly
disturbed wall). Therefore:
  - AW5D residual sigma  =  caliper transducer noise  +  drilling-induced
                            roughness
  - AW5O residual sigma  =  caliper transducer noise  (essentially)

By using the SAME detrending procedure (centred 0.30 m moving average)
on both wells, the difference between the two sigmas is interpretable
as the contribution of the drilling method.

Intervals (chosen for AW5O availability and AW5D legacy continuity)
  AW5O : [-5, -1] m          (most stable section of the AW5O log)
  AW5D : [-20, -15] m        (legacy interval used in earlier work)

Output
------
  notebooks\sandbox\07_caliper_analysis\fiona8\outputs\noise_comparison.json
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


MASTER_CSV = Path(r"data\caliper\concatenate_caliper_all.csv")
OUT_DIR = Path(r"notebooks\sandbox\07_caliper_analysis\fiona8\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETREND_WINDOW_M = 0.30


def moving_average_centered(y, win_pts):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if win_pts < 2 or n < 2:
        return y.copy()
    if win_pts % 2 == 0:
        win_pts += 1
    half = win_pts // 2
    pad_left = y[half:0:-1] if half > 0 else y[:0]
    pad_right = y[-2:-half - 2:-1] if half > 0 else y[:0]
    padded = np.concatenate([pad_left, y, pad_right])
    kernel = np.ones(win_pts) / win_pts
    return np.convolve(padded, kernel, mode="valid")[:n]


def lag1_autocorr(r):
    r = r - r.mean()
    return float(np.sum(r[:-1] * r[1:]) / np.sum(r ** 2))


def measure_noise(z, cal, z_lo, z_hi, detrend_win_m):
    mask = (z >= z_lo) & (z <= z_hi)
    z_sel, cal_sel = z[mask], cal[mask]
    dz = float(np.median(np.diff(z_sel)))
    win_pts = max(3, int(round(detrend_win_m / dz)))
    if win_pts % 2 == 0:
        win_pts += 1
    trend = moving_average_centered(cal_sel, win_pts)
    resid = cal_sel - trend
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return dict(
        well_interval=[z_lo, z_hi],
        n=int(len(z_sel)),
        median_dz_m=dz,
        detrend_window_m=detrend_win_m,
        detrend_window_pts=int(win_pts),
        cal_mean_cm=float(np.mean(cal_sel)),
        cal_std_cm=float(np.std(cal_sel, ddof=1)),
        sigma_std_cm=float(np.std(resid, ddof=1)),
        sigma_MAD_cm=1.4826 * mad,
        residual_mean_cm=float(np.mean(resid)),
        lag1_autocorr=lag1_autocorr(resid),
    )


df = pd.read_csv(MASTER_CSV)
df["well"] = df["source_file"].str.split("_").str[0]

print("=" * 78)
print("INSTRUMENTAL NOISE — comparable estimates AW5O vs AW5D")
print("=" * 78)
print(f"\nProcedure: centred {DETREND_WINDOW_M:.2f}-m moving-average detrending\n")

results = {}
for well, source, z_lo, z_hi in [
    ("AW5O", "AW5O_caliper_20210910.LAS", -5.0, -1.0),
    ("AW5D", "AW5D_caliper_20210910.LAS", -20.0, -15.0),
]:
    sub = df[df["source_file"] == source].sort_values("Depth [m]")
    z = sub["Depth [m]"].to_numpy()
    cal = sub["calibrated_cm"].to_numpy()
    auger_in = float(sub["Diameter_auger_in"].iloc[0])
    auger_cm = auger_in * 2.54

    res = measure_noise(z, cal, z_lo, z_hi, DETREND_WINDOW_M)
    res["well"] = well
    res["auger_in"] = auger_in
    res["auger_cm"] = auger_cm
    res["drilling_method"] = ("sonic_coring" if well == "AW5O"
                                else "rotary_auger")
    results[well] = res

    print(f"{well}  ({res['drilling_method']}, {auger_in:.0f}\" auger):")
    print(f"  Interval     : [{z_lo:+.0f}, {z_hi:+.0f}] m")
    print(f"  n samples    : {res['n']}")
    print(f"  cal mean     : {res['cal_mean_cm']:.3f} cm")
    print(f"  sigma_std    : {res['sigma_std_cm']:.4f} cm   <-- standard deviation of residuals")
    print(f"  sigma_MAD    : {res['sigma_MAD_cm']:.4f} cm   <-- robust (1.4826 * MAD)")
    print(f"  resid mean   : {res['residual_mean_cm']:+.5f} cm  (~0 expected)")
    print(f"  lag-1 autoc. : {res['lag1_autocorr']:+.3f}  (white noise = 0)")
    print()

# ── Comparison ─────────────────────────────────────────────────────────────
o = results["AW5O"]
d = results["AW5D"]
print("=" * 78)
print("COMPARISON (drilling-method contribution)")
print("=" * 78)
print(f"\n  sigma_std :  AW5D = {d['sigma_std_cm']:.4f} cm   "
      f"AW5O = {o['sigma_std_cm']:.4f} cm")
print(f"               difference = {d['sigma_std_cm'] - o['sigma_std_cm']:+.4f} cm  "
      f"(AW5D - AW5O)")
print(f"               ratio AW5D/AW5O = {d['sigma_std_cm']/o['sigma_std_cm']:.2f}x")
print(f"\n  sigma_MAD :  AW5D = {d['sigma_MAD_cm']:.4f} cm   "
      f"AW5O = {o['sigma_MAD_cm']:.4f} cm")
print(f"               difference = {d['sigma_MAD_cm'] - o['sigma_MAD_cm']:+.4f} cm  "
      f"(AW5D - AW5O)")
print(f"               ratio AW5D/AW5O = {d['sigma_MAD_cm']/o['sigma_MAD_cm']:.2f}x")

# Variance decomposition (assumes the two contributions add as independent
# Gaussian variances): sigma_AW5D^2 = sigma_inst^2 + sigma_drilling^2
# => sigma_drilling = sqrt(sigma_AW5D^2 - sigma_AW5O^2)
sd_drill_std = np.sqrt(max(0.0,
    d["sigma_std_cm"]**2 - o["sigma_std_cm"]**2))
sd_drill_mad = np.sqrt(max(0.0,
    d["sigma_MAD_cm"]**2 - o["sigma_MAD_cm"]**2))

print(f"\nIf the two contributions add as independent Gaussian variances:")
print(f"  sigma_drilling (from std) = sqrt(AW5D^2 - AW5O^2) = {sd_drill_std:.4f} cm")
print(f"  sigma_drilling (from MAD) = sqrt(AW5D^2 - AW5O^2) = {sd_drill_mad:.4f} cm")
print()
print("  Interpretation:")
print(f"    sigma_inst (caliper transducer)  ~  {o['sigma_std_cm']:.3f} cm  "
      f"(MAD: {o['sigma_MAD_cm']:.3f})")
print(f"    sigma_drilling (rotary auger)    ~  {sd_drill_std:.3f} cm  "
      f"(MAD: {sd_drill_mad:.3f})")
print()

results["comparison"] = dict(
    sigma_drilling_from_std_cm=sd_drill_std,
    sigma_drilling_from_MAD_cm=sd_drill_mad,
    interpretation=(
        "AW5O sigma is dominated by caliper transducer noise (smooth "
        "sonic-cored hole). AW5D sigma is the quadrature sum of "
        "transducer noise and rotary-auger drilling roughness."
    ),
)

with open(OUT_DIR / "noise_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT_DIR / 'noise_comparison.json'}")
