"""Vertical-reference transformations between instrument datums.

The YSI probe records depth below the water table (positive downward,
zero at the air-water interface inside the well). The caliper log
records depth below ground level (positive downward, zero at the
borehole collar). To compare both in the same well, the YSI depths
must be shifted by the vadose-zone thickness.

This module is intentionally minimal — only the operations needed for
this thesis week. Absolute-elevation conversions (between wells, between
sites) will be added later when the elevation-MSL dataset is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Default location of the well metadata table within the project.
DEFAULT_METADATA_PATH = Path("data/metadata/wells.csv")


def load_well_metadata(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the well metadata table.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. If None, ``data/metadata/wells.csv`` is used.

    Returns
    -------
    pd.DataFrame
        Columns include at least ``site``, ``well_type``,
        ``vadose_thickness_m``. Indexed by ``well_id`` (site + well_type)
        for fast lookup.
    """
    csv_path = Path(path) if path is not None else DEFAULT_METADATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Well metadata not found at '{csv_path}'. "
            f"Expected columns: site, well_type, vadose_thickness_m."
        )

    df = pd.read_csv(csv_path)

    required = {"site", "well_type", "vadose_thickness_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Well metadata at '{csv_path}' is missing columns: {sorted(missing)}"
        )

    # Build a `well_id` column for indexing (e.g. "AW6" + "D" -> "AW6D").
    df["well_id"] = df["site"].astype(str) + df["well_type"].astype(str)
    df = df.set_index("well_id")
    return df


def ysi_to_depth_below_ground(
    depth_ysi_m: np.ndarray | pd.Series,
    vadose_thickness_m: float,
) -> np.ndarray:
    """Convert YSI depth (below water table) to depth below ground level.

    Parameters
    ----------
    depth_ysi_m : array-like
        Depth values from the YSI instrument (positive downward, zero at
        the water-table contact inside the well).
    vadose_thickness_m : float
        Vadose-zone thickness in metres for the well being processed.
        That is, the distance from ground level down to the water table.

    Returns
    -------
    np.ndarray
        Depth below ground level, in metres, positive downward. The
        zero of the new axis is at the borehole collar (ground level).
    """
    if vadose_thickness_m < 0:
        raise ValueError(
            f"vadose_thickness_m must be non-negative; got {vadose_thickness_m}"
        )

    z = np.asarray(depth_ysi_m, dtype=float)
    return z + float(vadose_thickness_m)


def get_vadose_thickness(well_id: str, metadata: Optional[pd.DataFrame] = None) -> float:
    """Look up the vadose-zone thickness for a given well.

    Parameters
    ----------
    well_id : str
        Combined site + well_type identifier, e.g. ``"AW6D"``.
    metadata : pd.DataFrame, optional
        Pre-loaded metadata table. If None, loads from default path.

    Returns
    -------
    float
        Vadose-zone thickness in metres.

    Raises
    ------
    KeyError
        If the well_id is not found in the metadata table.
    """
    if metadata is None:
        metadata = load_well_metadata()

    if well_id not in metadata.index:
        raise KeyError(
            f"Well '{well_id}' not found in metadata. "
            f"Available wells: {metadata.index.tolist()}"
        )

    return float(metadata.loc[well_id, "vadose_thickness_m"])


# ──────────────────────────────────────────────────────────────────────
#  Vadose-zone thickness extraction from a raw YSI CSV
# ──────────────────────────────────────────────────────────────────────
# Some YSI exports include a column already referenced to ground level
# (``Depth from GL (m)``) alongside the probe's water-table-zero
# tracking column (``Vertical Position m``). When both are present,
# the vadose-zone thickness is the (constant) offset between them:
#
#     vadose_thickness_m = (Depth from GL) − (Vertical Position)
#
# This is the canonical operation that populated ``data/metadata/wells.csv``
# for the Feb-2022 priority wells. The driver script
# ``scripts/extract_vadose_from_raw.py`` orchestrates this over a folder
# tree; the function below is the single-file core operation that it
# (and any other caller) reuses.

DEPTH_FROM_GL_VARIANTS = [
    "depth from gl (m)",
    "depth from gl m",
    "depth_from_gl",
    "depth_bgl_m",
    "depth_bgl",
]
"""Header variants that mean *depth referenced to ground level*.

Match is case-insensitive and trimmed. Order matters only for clarity;
the lookup walks the list in order and returns the first hit."""

VERTICAL_POSITION_VARIANTS = [
    "vertical position [m]",
    "vertical position m",
    "vertical position",
    "depth_m",
    "vp",
]
"""Header variants that mean *depth referenced to the water table*.

Note: the YSI ``Depth m`` column is intentionally NOT in this list.
``Depth m`` and ``Vertical Position m`` look interchangeable but
differ by a small mechanical offset that the operator enters at the
surface, so subtracting ``Depth m`` from ``Depth from GL (m)`` yields
a non-constant offset across the cast. ``Vertical Position m`` is the
correct anchor — its offset is rigorously constant."""


def _find_col(df: "pd.DataFrame", variants: list[str]) -> Optional[str]:
    """Return the actual column name in ``df`` that matches any variant."""
    lower_to_actual = {c.strip().lower(): c for c in df.columns}
    for v in variants:
        actual = lower_to_actual.get(v.lower())
        if actual is not None:
            return actual
    return None


def extract_vadose_from_ysi_csv(
    csv_path: str | Path,
    *,
    std_tolerance_m: float = 0.01,
) -> tuple[Optional[float], str]:
    """Extract the vadose-zone offset from a single raw YSI CSV.

    The function inspects the CSV's column headers, computes
    ``(Depth from GL) − (Vertical Position)`` row-wise when both columns
    are present, and returns the median offset together with a status
    string describing what happened.

    Parameters
    ----------
    csv_path : str or Path
        Path to a raw YSI CSV.
    std_tolerance_m : float, default 0.01
        Maximum allowed standard deviation of the row-wise offset across
        a single profile. The offset should be rigorously constant on a
        healthy cast; a deviation above this tolerance is flagged in the
        status (but the median is still returned, so the caller can
        decide whether to use it).

    Returns
    -------
    (offset_m, status) : tuple
        offset_m : float or None
            Median vadose offset in metres, or None if either column is
            missing in the source CSV. The function never raises on a
            missing column — not all YSI exports include
            ``Depth from GL``, and the caller may want to fall back to a
            manual value.
        status : str
            One of:
                * ``"ok"`` — both columns found, std within tolerance.
                * ``"no_gl_column"`` — ``Depth from GL`` not found.
                * ``"no_vp_column"`` — ``Vertical Position`` not found.
                * ``"inconsistent (std=<value> m)"`` — std exceeds tolerance.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"YSI CSV not found: {path}")

    df = pd.read_csv(path)

    gl_col = _find_col(df, DEPTH_FROM_GL_VARIANTS)
    vp_col = _find_col(df, VERTICAL_POSITION_VARIANTS)

    if gl_col is None:
        return None, "no_gl_column"
    if vp_col is None:
        return None, "no_vp_column"

    diff = df[gl_col].astype(float) - df[vp_col].astype(float)
    diff = diff.dropna()
    if len(diff) == 0:
        return None, "inconsistent (no valid rows)"

    median_offset = float(np.median(diff))
    offset_std = float(np.std(diff))

    if offset_std > std_tolerance_m:
        # Still return the median so the caller can use it if they want,
        # but flag the fact that the offset is not rigorously constant.
        return median_offset, f"inconsistent (std={offset_std:.4f} m)"

    return median_offset, "ok"

