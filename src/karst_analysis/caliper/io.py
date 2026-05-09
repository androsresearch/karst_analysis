"""Loaders for caliper artefacts.

Two on-disk shapes are supported:

    * Master concatenated CSV (input to the pipeline) — produced upstream
      from individual LAS files. Columns: ``source_file``, ``Depth [m]``,
      ``calibrated_cm``, ``Diameter_auger_in``, plus a derived ``well``
      column added on load.

    * Per-sample CSV (output of the pipeline) — one row per caliper sample
      with the baseline, threshold, excess, severity and zone classification.
      Columns: ``well``, ``depth_m``, ``caliper_cm``, ``baseline_cm``,
      ``threshold_cm``, ``excess_from_threshold_cm``,
      ``severity_per_sample``, ``zone_label``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# Default locations within the project. Both are relative to the project
# root (the directory you run ``uv run`` from).
DEFAULT_MASTER_CSV = Path("data/raw/caliper/concatenate_caliper_all.csv")
DEFAULT_PERPOINT_CSV = Path("data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv")


def load_master_caliper(
    path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load the master concatenated caliper CSV.

    A ``well`` column is added (extracted as the first underscore-separated
    token of ``source_file``, e.g. ``"AW5D_caliper_20210910.LAS"`` →
    ``"AW5D"``).

    The on-disk depth column ``"Depth [m]"`` is renamed to ``depth_m``
    so it follows snake_case throughout the package, and is asserted
    to be in BGL-positive convention (matching the original LAS
    files). If you find the depths are negative, run
    ``scripts/fix_caliper_master_signs.py`` once to flip them.

    Parameters
    ----------
    path : str or Path, optional
        Override the default master CSV location.

    Returns
    -------
    pd.DataFrame
        Includes ``depth_m`` (BGL positive), ``calibrated_cm``,
        ``Diameter_auger_in``, ``source_file``, ``well``.
    """
    csv_path = Path(path) if path is not None else DEFAULT_MASTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Master caliper CSV not found: {csv_path}. "
            f"Expected columns: source_file, Depth [m], calibrated_cm, "
            f"Diameter_auger_in."
        )
    df = pd.read_csv(csv_path)

    required = {"source_file", "Depth [m]", "calibrated_cm", "Diameter_auger_in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Master caliper CSV at {csv_path} is missing columns: {sorted(missing)}."
        )

    df = df.rename(columns={"Depth [m]": "depth_m"}).copy()
    df["well"] = df["source_file"].str.split("_").str[0]

    # Sanity check: BGL-positive convention
    if df["depth_m"].min() < -0.1:
        raise ValueError(
            f"Master caliper CSV at {csv_path} has negative depths "
            f"(min={df['depth_m'].min():.3f}). The package now expects "
            f"BGL-positive depths. Run scripts/fix_caliper_master_signs.py "
            f"once to fix the file."
        )

    return df


def load_perpoint(
    path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load the per-sample CSV produced by the caliper pipeline.

    Used by downstream consumers (e.g. the convergence panel that
    overlays caliper, video and SEC) that don't need to re-run the
    detection pipeline themselves.

    Parameters
    ----------
    path : str or Path, optional
        Override the default per-sample CSV location.
    """
    csv_path = Path(path) if path is not None else DEFAULT_PERPOINT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-sample CSV not found: {csv_path}. "
            f"Run scripts/caliper_run_pipeline.py to generate it."
        )
    df = pd.read_csv(csv_path)

    required = {"well", "depth_m", "caliper_cm", "baseline_cm",
                "threshold_cm", "excess_from_threshold_cm",
                "severity_per_sample", "zone_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Per-sample CSV at {csv_path} is missing columns: {sorted(missing)}."
        )
    return df
