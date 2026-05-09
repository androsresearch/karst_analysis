"""Breakout detection over the cumulative-min baseline.

The detection rule, severity binning, and per-sample classification all
live here. The numerical primitive is a contiguous-runs filter that
keeps only sustained excursions above the threshold curve.

Migration history
-----------------
v5: extracted from ``cumulative_min_baseline.py`` (now
``karst_analysis.caliper.baseline``) with no algorithmic changes. The
byte-for-byte equivalence of the migrated pipeline against the original
is enforced by ``tests/test_caliper_*``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# =============================================================================

def _runs_of_true(flag: np.ndarray, min_len_pts: int) -> list[tuple[int, int]]:
    out, in_run, s = [], False, 0
    for i, f in enumerate(flag):
        if f and not in_run:
            s, in_run = i, True
        elif (not f) and in_run:
            e = i - 1
            if e - s + 1 >= min_len_pts:
                out.append((s, e))
            in_run = False
    if in_run and (len(flag) - 1 - s + 1) >= min_len_pts:
        out.append((s, len(flag) - 1))
    return out


def detect_breakouts_cumulative_min(
    z: np.ndarray,
    cal: np.ndarray,
    baseline: np.ndarray,
    offset_cm: float,
    sigma_inst_cm: float,
    k_sigma: float,
    L_min_m: float,
    saturation_cm: float,
    mild_max_excess_cm: float,
    moderate_max_excess_cm: float,
    nominal_cm: Optional[float] = None,
    zone_label: Optional[np.ndarray] = None,
) -> tuple[list[dict], dict]:
    """
    Detection rule:
        threshold(z) = baseline(z) + offset_cm + k_sigma * sigma_inst_cm
        breakout zone = run of >= L_min_m where caliper(z) > threshold(z)

    Severity (READING 2 — excess measured from THRESHOLD, not baseline):
        excess_over_thr(z) = caliper(z) - (baseline(z) + offset_cm)
                             [we ignore the noise term in the bin definition
                              because k*sigma is tiny vs the offset]

    Per-sample classification (used for the per-sample band plot):
        mild_pp     : 0 < excess_over_thr < mild_max_excess_cm
        moderate_pp : mild_max <= excess_over_thr < moderate_max_excess_cm
        severe_pp   : excess_over_thr >= moderate_max_excess_cm  OR
                      caliper >= saturation_cm

    Returns
    -------
    zones : list of dicts (each contiguous run above threshold), with the
            severity of its PEAK sample reported in zone['severity'].
    perpoint : dict with arrays aligned to z:
            'severity'        - one of 'none', 'mild', 'moderate', 'severe'
            'excess_over_thr' - cm above threshold (NaN where no baseline)
            'over_threshold'  - bool, the detection flag per sample
    """
    z = np.asarray(z, dtype=float)
    cal = np.asarray(cal, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    # Per-sample threshold and excess (from threshold)
    threshold_curve = baseline + offset_cm + k_sigma * sigma_inst_cm
    # The "excess" used for severity is measured from the offset boundary
    # (B + offset), not from the noise-corrected threshold, because the
    # cm-bins are about physical separation from B+offset, not statistical.
    excess_over_thr = cal - (baseline + offset_cm)

    flag = cal > threshold_curve
    if zone_label is not None:
        flag = flag & (zone_label != "excluded")
    flag = flag & np.isfinite(baseline)

    # Per-sample severity (only meaningful where flag is True)
    severity_per_pt = np.full(len(cal), "none", dtype=object)
    mask_sev_excess = flag & (excess_over_thr >= moderate_max_excess_cm)
    mask_sev_sat = flag & (cal >= saturation_cm)
    mask_sev = mask_sev_excess | mask_sev_sat
    mask_mod = flag & (excess_over_thr >= mild_max_excess_cm) & ~mask_sev
    mask_mild = flag & ~mask_mod & ~mask_sev
    severity_per_pt[mask_mild] = "mild"
    severity_per_pt[mask_mod] = "moderate"
    severity_per_pt[mask_sev] = "severe"

    perpoint = dict(
        severity=severity_per_pt,
        excess_over_thr=np.where(np.isfinite(threshold_curve),
                                   excess_over_thr, np.nan),
        over_threshold=flag,
        threshold_curve=threshold_curve,
    )

    # Zone runs (contiguous regions above threshold of length >= L_min)
    dz = float(np.median(np.diff(np.sort(z))))
    min_run_pts = max(1, int(round(L_min_m / abs(dz))))
    runs = _runs_of_true(flag, min_run_pts)

    zones = []
    for s, e in runs:
        seg_cal = cal[s:e + 1]
        seg_base = baseline[s:e + 1]
        seg_excess = excess_over_thr[s:e + 1]
        i_peak = int(np.argmax(seg_cal))
        peak_cm = float(seg_cal[i_peak])
        b_peak = float(seg_base[i_peak])
        peak_excess_baseline = peak_cm - b_peak              # legacy: from B
        peak_excess_thr = peak_cm - (b_peak + offset_cm)     # from threshold

        z_top = float(max(z[s], z[e]))
        z_bot = float(min(z[s], z[e]))
        n_sat = int((seg_cal >= saturation_cm).sum())

        # Zone-level severity: severity of the PEAK sample
        if n_sat > 0 or peak_excess_thr >= moderate_max_excess_cm:
            severity = "severe"
        elif peak_excess_thr >= mild_max_excess_cm:
            severity = "moderate"
        else:
            severity = "mild"

        # Per-sample severity counts inside this zone (informative)
        seg_sev = severity_per_pt[s:e + 1]
        n_pp_mild = int(np.sum(seg_sev == "mild"))
        n_pp_mod = int(np.sum(seg_sev == "moderate"))
        n_pp_sev = int(np.sum(seg_sev == "severe"))

        zone = dict(
            z_top=z_top,
            z_bot=z_bot,
            z_centre=0.5 * (z_top + z_bot),
            thickness_m=float(abs(z[e] - z[s])),
            peak_cm=peak_cm,
            baseline_at_peak_cm=b_peak,
            peak_excess_cm=peak_excess_baseline,        # kept for backward-compat
            peak_excess_over_threshold_cm=peak_excess_thr,
            n_saturated=n_sat,
            severity=severity,
            n_pp_mild=n_pp_mild,
            n_pp_moderate=n_pp_mod,
            n_pp_severe=n_pp_sev,
            offset_cm=offset_cm,
            k_sigma=k_sigma,
            sigma_inst_cm=sigma_inst_cm,
            mild_max_excess_cm=mild_max_excess_cm,
            moderate_max_excess_cm=moderate_max_excess_cm,
        )
        if nominal_cm is not None:
            zone["peak_excess_vs_nominal_cm"] = peak_cm - nominal_cm

        if zone_label is not None:
            labels_in_run = zone_label[s:e + 1]
            uniq = np.unique(labels_in_run)
            zone["sub_zone"] = "+".join(sorted(uniq.tolist()))
        zones.append(zone)
    return zones, perpoint
