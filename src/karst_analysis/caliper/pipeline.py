"""End-to-end caliper pipeline orchestration.

Combines the noise estimate, the cumulative-min baseline fit, and the
breakout detection into a single function that takes raw inputs and
returns the per-well per-sample results plus the zone-level summary.

The two CLI scripts ``scripts/caliper_estimate_noise.py`` and
``scripts/caliper_run_pipeline.py`` are thin wrappers over this module
plus :mod:`karst_analysis.caliper.noise`.

Migration history
-----------------
v5: extracted from ``priority_wells_cumulative_min_v2.py`` and
``export_perpoint_breakouts.py`` with no algorithmic changes. The
byte-for-byte equivalence of the migrated pipeline against the original
is enforced by ``tests/test_caliper_pipeline_e2e.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.caliper.baseline import fit_cumulative_min_split
from karst_analysis.caliper.detection import detect_breakouts_cumulative_min
from karst_analysis.caliper.config import (
    OFFSET_CM, K_SIGMA, L_MIN_M, SATURATION_CM,
    MILD_MAX_EXCESS_CM, MODERATE_MAX_EXCESS_CM,
    PRIORITY_WELLS, TRIM_DEPTHS_M,
    DEFAULT_INTERP_KIND, DEFAULT_DIRECTION, DEFAULT_IQR_K,
)


# ──────────────────────────────────────────────────────────────────────
#  Per-well processing
# ──────────────────────────────────────────────────────────────────────
def process_one_well(
    well_id: str,
    df_master: pd.DataFrame,
    sigma_inst_cm: float,
    *,
    trim_depth_m: Optional[float] = None,
    offset_cm: float = OFFSET_CM,
    k_sigma: float = K_SIGMA,
    l_min_m: float = L_MIN_M,
    saturation_cm: float = SATURATION_CM,
    mild_max_excess_cm: float = MILD_MAX_EXCESS_CM,
    moderate_max_excess_cm: float = MODERATE_MAX_EXCESS_CM,
    interp_kind: str = DEFAULT_INTERP_KIND,
    direction: str = DEFAULT_DIRECTION,
    iqr_k: float = DEFAULT_IQR_K,
) -> dict:
    """Run the full baseline-fit + detection on one well.

    Parameters
    ----------
    well_id : str
        Well identifier (e.g. ``"AW5D"``). Must be a value present in the
        ``well`` column of ``df_master``.
    df_master : pd.DataFrame
        Output of :func:`karst_analysis.caliper.io.load_master_caliper`.
    sigma_inst_cm : float
        Instrumental-noise estimate (cm). Typically the AW5O ``sigma_MAD``
        from the noise-comparison report.
    trim_depth_m : float, optional
        Boundary between shallow and deep zones. If None, looks up the
        well-specific value in
        :data:`karst_analysis.caliper.config.TRIM_DEPTHS_M`.

    Returns
    -------
    dict
        Keys: ``well``, ``z``, ``cal``, ``auger_in``, ``auger_cm``, ``fit``,
        ``zones``, ``perpoint``, ``trim_depth_m``, ``n_below_auger``.
    """
    if trim_depth_m is None:
        if well_id not in TRIM_DEPTHS_M:
            raise KeyError(
                f"No default trim_depth_m for {well_id}; "
                f"add it to caliper.config.TRIM_DEPTHS_M or pass trim_depth_m="
            )
        trim_depth_m = TRIM_DEPTHS_M[well_id]

    sub = df_master[df_master["well"] == well_id].copy()
    if sub.empty:
        raise ValueError(f"No rows for well_id={well_id} in df_master.")

    sub = sub.sort_values("depth_m").reset_index(drop=True)
    z = sub["depth_m"].to_numpy()
    cal = sub["calibrated_cm"].to_numpy()
    auger_in = float(sub["Diameter_auger_in"].iloc[0])
    auger_cm = auger_in * 2.54

    fit = fit_cumulative_min_split(
        z, cal,
        trim_depth_m=trim_depth_m,
        interp_kind=interp_kind,
        direction=direction,
        analyse_shallow=True,
        floor_cm=auger_cm,
        iqr_k=iqr_k,
    )
    zones, perpoint = detect_breakouts_cumulative_min(
        z, cal, fit.baseline,
        offset_cm=offset_cm,
        sigma_inst_cm=sigma_inst_cm, k_sigma=k_sigma,
        L_min_m=l_min_m,
        saturation_cm=saturation_cm,
        mild_max_excess_cm=mild_max_excess_cm,
        moderate_max_excess_cm=moderate_max_excess_cm,
        nominal_cm=auger_cm,
        zone_label=fit.zone_label,
    )

    n_below_auger = int((cal < auger_cm).sum())

    return dict(
        well=well_id,
        z=z, cal=cal,
        auger_in=auger_in, auger_cm=auger_cm,
        fit=fit, zones=zones, perpoint=perpoint,
        trim_depth_m=trim_depth_m,
        n_below_auger=n_below_auger,
    )


# ──────────────────────────────────────────────────────────────────────
#  Multi-well batch and serialisation helpers
# ──────────────────────────────────────────────────────────────────────
def process_many_wells(
    df_master: pd.DataFrame,
    sigma_inst_cm: float,
    *,
    wells: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, dict]:
    """Run :func:`process_one_well` for each well in ``wells``.

    If ``wells`` is None, defaults to
    :data:`karst_analysis.caliper.config.PRIORITY_WELLS`.

    Returns
    -------
    dict
        ``{well_id: result_dict}``.
    """
    if wells is None:
        wells = PRIORITY_WELLS
    return {w: process_one_well(w, df_master, sigma_inst_cm, **kwargs) for w in wells}


def perpoint_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Build the per-sample CSV from a multi-well result dict.

    Schema matches the original ``priority_wells_cumulative_min_v2_perpoint.csv``:

        well, depth_m, caliper_cm, baseline_cm, threshold_cm,
        excess_from_threshold_cm, severity_per_sample, zone_label
    """
    frames = []
    for well_id, r in results.items():
        z = r["z"]
        cal = r["cal"]
        baseline = r["fit"].baseline
        threshold = r["perpoint"]["threshold_curve"]
        excess_from_threshold = cal - threshold
        severity = r["perpoint"]["severity"]
        zone_label = r["fit"].zone_label

        frames.append(pd.DataFrame({
            "well": well_id,
            "depth_m": z,
            "caliper_cm": cal,
            "baseline_cm": baseline,
            "threshold_cm": threshold,
            "excess_from_threshold_cm": excess_from_threshold,
            "severity_per_sample": severity,
            "zone_label": zone_label,
        }))
    cols = ["well", "depth_m", "caliper_cm", "baseline_cm", "threshold_cm",
            "excess_from_threshold_cm", "severity_per_sample", "zone_label"]
    return pd.concat(frames, ignore_index=True)[cols]


def zones_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Build the zone-level CSV from a multi-well result dict.

    Schema matches the original ``priority_wells_cumulative_min_v2_zones.csv``.
    """
    rows = []
    for well_id, r in results.items():
        for zn in r["zones"]:
            rows.append({"well": well_id, **zn})
    return pd.DataFrame(rows)
