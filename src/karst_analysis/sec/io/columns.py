"""Column name conventions for SEC CSV files.

YSI exports vary between firmware versions and between operators, so
columns of interest (depth, conductivity, time) appear under several
slightly different names. This module centralises the mapping.

After auto-detection, internal code uses standardised names:
    depth_m         (originally "Vertical Position [m]" etc.)
    sec_uS_cm       (originally "Corrected sp Cond [µS/cm]" etc.)

The standardisation step is optional — most existing functions accept
either the original or the standardised names via :func:`find_column_name`.

Notes for v12
-------------
* Some 2025-era YSI exports replace ``µ`` with ``mu`` in column names
  (``SpCond_muS/cm`` instead of ``SpCond µS/cm``); the new aliases are
  included below.
* Some 2025-era exports also dropped ``Vertical Position m`` entirely
  and only ship ``Depth m``. ``Depth m`` is included as a LAST-RESORT
  alias for the depth concept — it is functionally equivalent to
  ``Vertical Position m`` for plotting, but lacks the small mechanical
  offset that ``Vertical Position m`` carries. For vadose-thickness
  EXTRACTION (``corrections.datum.extract_vadose_from_ysi_csv``), we
  STILL prefer ``Vertical Position m`` because the offset to ground
  level is rigorously constant only with that column.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# Known variants for each conceptual column. Order matters: first match wins.
DEFAULT_COLUMN_MAPPINGS: dict[str, list[str]] = {
    "depth": [
        "depth_m",
        "Vertical Position [m]",
        "Vertical Position m",
        "VP",
        "z",
        "Z",
        # Last-resort fallback: when 'Vertical Position m' is absent (some
        # 2025-era exports) we accept 'Depth m' for plotting purposes.
        "Depth m",
    ],
    "conductivity": [
        "sec_uS_cm",
        "Corrected sp Cond [uS/cm]",
        "Corrected sp Cond [µS/cm]",
        "SpCond_muS/cm",
        "SpCond µS/cm",
        "SEC",
        "Conductivity",
        "conductivity",
        "EC",
        "ec",
    ],
    "time": [
        "Time (HH:mm:ss)",
        "Time (HH:MM:SS)",  # 2025_02 firmware capitalises the hour fields
    ],
    "time_frac": [
        "Time (Fract. Sec)",
    ],
}


# Canonical names used internally after standardisation.
STANDARD_NAMES = {
    "depth": "depth_m",
    "conductivity": "sec_uS_cm",
}


def find_column_name(
    df: pd.DataFrame,
    column_type: str,
    column_mappings: Optional[dict[str, list[str]]] = None,
) -> Optional[str]:
    """Return the actual column name in ``df`` for a given concept.

    Parameters
    ----------
    df : pd.DataFrame
    column_type : str
        One of the keys in ``column_mappings`` (e.g. ``"depth"``,
        ``"conductivity"``, ``"time"``).
    column_mappings : dict, optional
        Custom mapping. If None, uses :data:`DEFAULT_COLUMN_MAPPINGS`.

    Returns
    -------
    str or None
        The matching column name, or None if nothing matched.
    """
    if column_mappings is None:
        column_mappings = DEFAULT_COLUMN_MAPPINGS

    candidates = column_mappings.get(column_type, [])

    # Exact match first.
    for name in candidates:
        if name in df.columns:
            return name

    # Case-insensitive fallback.
    lower_cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        actual = lower_cols.get(name.lower())
        if actual is not None:
            return actual

    return None


def standardise_columns(
    df: pd.DataFrame,
    column_mappings: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Rename detected columns to the project's standard names.

    Returns a copy with renamed columns; original columns not in the
    mapping are preserved untouched.

    Standard names produced (when source columns are detected):
        depth        → "depth_m"
        conductivity → "sec_uS_cm"

    If a concept is not found, the corresponding rename is skipped.
    """
    if column_mappings is None:
        column_mappings = DEFAULT_COLUMN_MAPPINGS

    rename_map = {}
    for concept, std_name in STANDARD_NAMES.items():
        found = find_column_name(df, concept, column_mappings)
        if found is not None and found != std_name:
            rename_map[found] = std_name

    return df.rename(columns=rename_map)
