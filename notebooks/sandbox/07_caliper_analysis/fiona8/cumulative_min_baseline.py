"""
cumulative_min_baseline.py — Supervisor's running-cumulative-minimum baseline
============================================================================

This module implements the baseline-construction method proposed by the
supervisor (handwritten note + Excel sketch), and adapts it to handle the
shallow drilling-disturbed zone separately from the rest of the well.

THE METHOD
----------
Walk the caliper log from the surface downward, sample by sample. Maintain
a running record of the smallest caliper value seen so far. The baseline
M(z) at any depth z is, by definition, the smallest caliper recorded between
the top of the log and z:

    M(z) = min{ C(z') : z' is shallower than or equal to z }

By construction:
    (i)   M(z) is monotonically NON-INCREASING with depth (it can only
          drop or stay flat, never rise);
    (ii)  M(z) <= C(z) at every z, so the local excess  C(z) - M(z)  is
          non-negative;
    (iii) M(z) equals C(z) at "anchor depths" — the points where the log
          set a new shallowest-yet minimum.

Detection rule (from the same handwritten note):

    breakout(z) = True   iff   C(z)  >  M(z) * (1 + alpha) + k * sigma

where sigma is the instrumental noise (measured on a "really smooth"
reference well, e.g. AW5O) and k is a coverage factor (typical values:
3 for "3-sigma", or 0 to drop the additive guard entirely).

WHY THIS WORKS WHERE THE ENVELOPE METHOD HAD TROUBLE
----------------------------------------------------
The supervisor's critique of the running-min-plus-smooth envelope was that
in the SHALLOW zone (where the auger plays mechanically loose, and the
hole is genuinely wider than at depth), the envelope drops to follow the
local minima. Anything above that local minimum is then flagged as a
"breakout", which floods the shallow zone with false positives.

The cumulative minimum is different: it can only drop, never rise. So once
M(z) latches onto a narrow value somewhere in the log, every shallower
point inherits that narrow reference. In particular, if the LOG STARTS
wide (because of shallow over-reaming), but reaches a narrow value soon
after, the shallow zone's threshold is set by the SHALLOW PART of the log
itself (because M is non-increasing — the shallow values cannot benefit
from narrow values found below them later). This is fine IF the user is
willing to accept that the shallow zone uses its own local reference.

The trick — and what part (b) of the user's request implements — is to
ANALYSE THE TWO ZONES SEPARATELY:

    Shallow zone:  z >= trim_depth_well   (typically the topmost 1-3 m,
                                           drilling-disturbed)
    Deep zone:     z <  trim_depth_well   (the rest of the well)

Inside each zone we compute its OWN cumulative minimum (each zone starts
with its own running-min state). This keeps the deep-zone baseline
unaffected by shallow over-reaming, and gives the shallow zone its own
internal reference rather than inheriting a deep-only minimum that would
make every shallow point look anomalous.

INTERPOLATION BETWEEN ANCHOR POINTS
-----------------------------------
The cumulative-min sequence M(z) is piecewise constant — a staircase that
steps down at each new minimum and stays flat in between. This is the most
faithful representation of the method ("the minimum value seen up to depth
z is a STEP function"). For visualisation or smoother thresholds we also
provide:
    - 'linear' interpolation between consecutive anchor points, and
    - 'pchip' (Piecewise Cubic Hermite, monotone) interpolation, which is
      smooth and preserves monotonicity.

The default 'step' is recommended for analysis: it does not invent
information between anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator


# =============================================================================
#  RESULT CONTAINER
# =============================================================================

@dataclass
class CumulativeMinResult:
    """Result of cumulative-minimum baseline fit on a single zone."""
    z: np.ndarray              # depth array (full input, sorted top-to-bottom)
    cal: np.ndarray            # caliper input, aligned with z
    baseline: np.ndarray       # M(z), same length as z
    anchor_indices: np.ndarray # indices in z where a new minimum was set
    anchor_z: np.ndarray       # depths of the anchor points
    anchor_cal: np.ndarray     # caliper values at anchor points
    interp_kind: str           # 'step', 'linear', or 'pchip'

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_indices)


# =============================================================================
#  CORE CUMULATIVE-MIN PRIMITIVE
# =============================================================================

def _running_cumulative_minimum(values: np.ndarray) -> np.ndarray:
    """np.minimum.accumulate gives the prefix minimum; that is exactly the
    'smallest value seen so far' iterating left-to-right."""
    return np.minimum.accumulate(values)


def _find_anchors(values: np.ndarray) -> np.ndarray:
    """Return the indices where values[i] is a NEW strict minimum — i.e.
    smaller than every value at indices < i. The first index is always an
    anchor (no prior values). The supervisor's note says:

        if caliper_z < caliper_{z-1}  then  write (z, caliper_z) to file
        else                              do not write to file

    However, taken literally that rule produces an anchor only when the
    SAMPLE drops below its IMMEDIATE PREVIOUS sample, which is not the
    same thing as a new running-minimum: noise can produce many such
    "drops" that are not actually new minima. The Excel plot makes clear
    that what she actually wants is new RUNNING minima (the staircase
    only steps down to values smaller than ALL previous values), so we
    implement that.
    """
    cum_min = _running_cumulative_minimum(values)
    # An anchor is a position where the cumulative min strictly drops,
    # plus the very first sample.
    is_anchor = np.concatenate(([True], np.diff(cum_min) < 0))
    return np.flatnonzero(is_anchor)


# =============================================================================
#  SINGLE-ZONE FIT
# =============================================================================

def fit_cumulative_min_single_zone(
    z: np.ndarray,
    cal: np.ndarray,
    interp_kind: Literal["step", "linear", "pchip"] = "step",
    direction: Literal["top_down", "bottom_up"] = "top_down",
) -> CumulativeMinResult:
    """
    Fit a cumulative-minimum baseline on one contiguous zone.

    Parameters
    ----------
    z, cal : 1-D arrays of equal length
        Depth (in metres, may be negative) and caliper (cm). Will be
        sorted internally; the ORIGINAL ordering of (z, cal) is preserved
        in the returned `z` and `cal`.
    interp_kind : 'step', 'linear', or 'pchip'
        How to fill the baseline between consecutive anchor points.
        'step' is the strictest reading of the supervisor's method.
    direction : 'top_down' or 'bottom_up'
        Direction of the cumulative-min walk. 'top_down' (default) is what
        the supervisor described: surface-to-bottom; the shallow zone uses
        whatever narrow values appear first. 'bottom_up' walks from the
        bottom upward; useful as a sensitivity check or if one wants the
        baseline to be set by the narrowest values reached anywhere below
        a given depth.

    Returns
    -------
    CumulativeMinResult
    """
    z = np.asarray(z, dtype=float)
    cal = np.asarray(cal, dtype=float)
    if z.shape != cal.shape:
        raise ValueError("z and cal must have the same length")
    if len(z) < 2:
        raise ValueError("Need at least 2 points to build a baseline")

    # Sort by depth, surface (z=0 or z<0 close to 0) at the top.
    # In this project z is negative (e.g. -35..-0.27), so 'surface' = LARGEST
    # z value. For the cumulative min to walk "top-down" we therefore iterate
    # in order of DECREASING z.
    sort_top_down = np.argsort(-z)         # surface first
    z_sorted = z[sort_top_down]
    cal_sorted = cal[sort_top_down]

    if direction == "top_down":
        walking_cal = cal_sorted
    elif direction == "bottom_up":
        walking_cal = cal_sorted[::-1]
    else:
        raise ValueError("direction must be 'top_down' or 'bottom_up'")

    cum_min = _running_cumulative_minimum(walking_cal)
    anchor_idx_walk = _find_anchors(walking_cal)

    if direction == "bottom_up":
        cum_min = cum_min[::-1]
        anchor_idx_walk = (len(walking_cal) - 1) - anchor_idx_walk[::-1]

    # `cum_min` is now in surface-first order.
    # Build interpolated baseline between anchors:
    if interp_kind == "step":
        baseline_surface_first = cum_min.copy()
    else:
        z_anch = z_sorted[anchor_idx_walk]
        cal_anch = cum_min[anchor_idx_walk]
        # Need anchors monotone in z (they are already, since cum_min picks
        # them in z-order). PCHIP and linear need ascending x:
        order_x = np.argsort(z_anch)
        xs = z_anch[order_x]
        ys = cal_anch[order_x]
        if interp_kind == "linear":
            baseline_surface_first = np.interp(z_sorted, xs, ys)
        elif interp_kind == "pchip":
            if len(xs) >= 2:
                pchip = PchipInterpolator(xs, ys, extrapolate=True)
                baseline_surface_first = pchip(z_sorted)
            else:
                baseline_surface_first = np.full_like(z_sorted, ys[0])
        else:
            raise ValueError("interp_kind must be 'step', 'linear', 'pchip'")

    # Map anchor indices from surface-first order back to ORIGINAL order
    # so that the returned arrays are aligned with the caller's (z, cal).
    anchor_idx_in_sorted = anchor_idx_walk
    inverse_perm = np.empty_like(sort_top_down)
    inverse_perm[sort_top_down] = np.arange(len(z))
    anchor_idx_orig = sort_top_down[anchor_idx_in_sorted]

    baseline_orig_order = np.empty_like(baseline_surface_first)
    baseline_orig_order[sort_top_down] = baseline_surface_first

    return CumulativeMinResult(
        z=z, cal=cal,
        baseline=baseline_orig_order,
        anchor_indices=anchor_idx_orig,
        anchor_z=z[anchor_idx_orig],
        anchor_cal=cal[anchor_idx_orig],
        interp_kind=interp_kind,
    )


# =============================================================================
#  TWO-ZONE FIT (shallow / deep), with INDEPENDENT cumulative mins
# =============================================================================

@dataclass
class SplitCumulativeMinResult:
    """Result of a two-zone (shallow + deep) cumulative-min baseline fit."""
    z: np.ndarray
    cal: np.ndarray
    baseline: np.ndarray              # combined baseline over the full log
    zone_label: np.ndarray            # 'shallow', 'deep', or 'excluded' per sample
    shallow: Optional[CumulativeMinResult]
    deep: Optional[CumulativeMinResult]
    trim_depth_m: float
    direction: str
    interp_kind: str


def fit_cumulative_min_split(
    z: np.ndarray,
    cal: np.ndarray,
    trim_depth_m: float,
    interp_kind: Literal["step", "linear", "pchip"] = "step",
    direction: Literal["top_down", "bottom_up"] = "top_down",
    analyse_shallow: bool = True,
    floor_cm: Optional[float] = None,
    iqr_k: Optional[float] = 1.5,
) -> SplitCumulativeMinResult:
    """
    Fit cumulative-minimum baselines independently on the shallow and deep
    zones, separated by `trim_depth_m`.

    Convention
    ----------
    Depth is given as a NEGATIVE elevation (e.g. -1.0 m means 1 m below
    surface). Therefore:
        shallow zone  =  z > trim_depth_m  (i.e. closer to the surface)
        deep zone     =  z <= trim_depth_m

    Setting analyse_shallow=False reproduces the legacy behaviour of just
    excluding the shallow zone from the analysis.

    Outlier handling
    ----------------
    The cumulative-min method is very sensitive to a single anomalously
    LOW caliper value: it would latch onto that value and drag the baseline
    too low for everything below it. Two filters are applied to the input
    BEFORE the cumulative min is computed:

      (a) `floor_cm`: any sample with caliper < floor_cm is rejected as
          physically impossible (a caliper cannot record an aperture
          smaller than the auger that drilled the hole). Pass the auger
          diameter as `floor_cm` to enforce this.

      (b) `iqr_k`: standard Tukey lower-fence outlier flag. Samples below
          Q1 - iqr_k*IQR are also rejected. Set iqr_k=None to disable.

    Rejected samples keep their position in the returned arrays — the
    baseline at those positions is filled by interpolation from the
    surrounding valid anchors (so the baseline remains defined for ALL
    input depths). Detection at rejected samples is still possible; their
    raw caliper is compared against the interpolated baseline + threshold.

    Returns
    -------
    SplitCumulativeMinResult
    """
    z = np.asarray(z, dtype=float)
    cal = np.asarray(cal, dtype=float)

    # Build the validity mask — points that are KEPT for the cumulative-min
    # computation. We do NOT drop them from the returned arrays.
    valid = np.ones(len(z), dtype=bool)
    if floor_cm is not None:
        valid &= cal >= floor_cm
    if iqr_k is not None and iqr_k > 0:
        q1 = np.nanpercentile(cal[valid], 25)
        q3 = np.nanpercentile(cal[valid], 75)
        iqr = q3 - q1
        lower_fence = q1 - iqr_k * iqr
        valid &= cal >= lower_fence

    is_shallow = z > trim_depth_m
    is_deep = ~is_shallow

    baseline = np.full_like(cal, np.nan)
    zone_label = np.empty(len(z), dtype=object)
    zone_label[:] = "excluded"

    shallow_res = None
    deep_res = None

    deep_valid = is_deep & valid
    if deep_valid.sum() >= 2:
        deep_res = fit_cumulative_min_single_zone(
            z[deep_valid], cal[deep_valid],
            interp_kind=interp_kind, direction=direction)
        # Map the result back to the full deep zone (including invalid
        # samples) by interpolating the deep baseline onto every deep z.
        # Use linear interpolation against the anchor points, which is
        # well-defined even if interp_kind is 'step' (we just want a value
        # at every z to allow detection).
        z_deep_all = z[is_deep]
        if interp_kind == "step":
            # For 'step' interpolation outside the fitted points we fall
            # back to the running cumulative min logic: at any z, the
            # baseline is the smallest VALID caliper seen between the top
            # of the deep zone and z (top-down direction).
            # Implement explicitly by walking the full deep zone with NaN
            # masking.
            order = np.argsort(-z_deep_all) if direction == "top_down" \
                    else np.argsort(z_deep_all)
            cal_deep_all = cal[is_deep][order]
            valid_deep_all = valid[is_deep][order]
            running = np.full(len(cal_deep_all), np.nan)
            current_min = np.nan
            for i, (c, v) in enumerate(zip(cal_deep_all, valid_deep_all)):
                if v:
                    if np.isnan(current_min) or c < current_min:
                        current_min = c
                running[i] = current_min
            # Forward-fill any leading NaNs (if first samples were invalid)
            if np.isnan(running[0]):
                first_valid = np.flatnonzero(~np.isnan(running))
                if len(first_valid):
                    running[:first_valid[0]] = running[first_valid[0]]
            # Unscramble back to original deep ordering
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            baseline[is_deep] = running[inv]
        else:
            # For 'linear' / 'pchip', extrapolate using the fitted curve
            xs = deep_res.anchor_z
            ys = deep_res.anchor_cal
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            if interp_kind == "linear":
                baseline[is_deep] = np.interp(z_deep_all, xs, ys)
            else:
                pchip = PchipInterpolator(xs, ys, extrapolate=True)
                baseline[is_deep] = pchip(z_deep_all)
        zone_label[is_deep] = "deep"

    if analyse_shallow:
        shallow_valid = is_shallow & valid
        if shallow_valid.sum() >= 2:
            shallow_res = fit_cumulative_min_single_zone(
                z[shallow_valid], cal[shallow_valid],
                interp_kind=interp_kind, direction=direction)
            z_sh_all = z[is_shallow]
            if interp_kind == "step":
                order = np.argsort(-z_sh_all) if direction == "top_down" \
                        else np.argsort(z_sh_all)
                cal_sh_all = cal[is_shallow][order]
                valid_sh_all = valid[is_shallow][order]
                running = np.full(len(cal_sh_all), np.nan)
                current_min = np.nan
                for i, (c, v) in enumerate(zip(cal_sh_all, valid_sh_all)):
                    if v:
                        if np.isnan(current_min) or c < current_min:
                            current_min = c
                    running[i] = current_min
                if np.isnan(running[0]):
                    first_valid = np.flatnonzero(~np.isnan(running))
                    if len(first_valid):
                        running[:first_valid[0]] = running[first_valid[0]]
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))
                baseline[is_shallow] = running[inv]
            else:
                xs = shallow_res.anchor_z
                ys = shallow_res.anchor_cal
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
                if interp_kind == "linear":
                    baseline[is_shallow] = np.interp(z_sh_all, xs, ys)
                else:
                    pchip = PchipInterpolator(xs, ys, extrapolate=True)
                    baseline[is_shallow] = pchip(z_sh_all)
            zone_label[is_shallow] = "shallow"

    return SplitCumulativeMinResult(
        z=z, cal=cal, baseline=baseline, zone_label=zone_label,
        shallow=shallow_res, deep=deep_res,
        trim_depth_m=trim_depth_m,
        direction=direction, interp_kind=interp_kind,
    )


# =============================================================================
#  DETECTION RULE
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
