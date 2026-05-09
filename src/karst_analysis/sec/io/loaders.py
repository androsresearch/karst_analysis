"""Unified loader for YSI SEC CSV files.

Combines the loading logic from the legacy ``modules/load.py`` and the
LOWESS pipeline ``load_profile``. Returns a tidy DataFrame with at least
the standardised columns ``depth_m`` and ``sec_uS_cm``.

Design choices:
    - Returns a DataFrame, not loose arrays. Downstream code can pull
      the columns it needs.
    - Conserves the original columns under their original names, plus
      adds the standardised aliases. This keeps the CSV self-describing
      and avoids losing time/index information.
    - Does NOT do any cleaning, smoothing, or unit conversion. Those are
      explicit pipeline steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from karst_analysis.sec.io.columns import (
    DEFAULT_COLUMN_MAPPINGS,
    find_column_name,
    standardise_columns,
)


def load_ysi_csv(
    filepath: str | Path,
    column_mappings: Optional[dict[str, list[str]]] = None,
    standardise: bool = True,
) -> pd.DataFrame:
    """Load a YSI SEC CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    column_mappings : dict, optional
        Override the default column-name mappings.
    standardise : bool, default True
        If True, rename the detected depth/conductivity columns to
        ``depth_m`` / ``sec_uS_cm``. If False, the DataFrame keeps the
        original column names (use :func:`find_column_name` to locate
        them).

    Returns
    -------
    pd.DataFrame
        The loaded data. Guaranteed to contain at least the standardised
        depth and conductivity columns when ``standardise=True``.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If neither a depth nor a conductivity column can be detected.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Validate that at least depth and conductivity exist.
    depth_col = find_column_name(df, "depth", column_mappings)
    sec_col = find_column_name(df, "conductivity", column_mappings)

    if depth_col is None:
        raise ValueError(
            f"No depth column found in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )
    if sec_col is None:
        raise ValueError(
            f"No conductivity column found in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if standardise:
        df = standardise_columns(df, column_mappings)

    return df


# ──────────────────────────────────────────────────────────────────────
#  Multi-trace loader (one well, one campaign, possibly several CSVs)
# ──────────────────────────────────────────────────────────────────────
import re
from dataclasses import dataclass, field
from typing import List

from karst_analysis.corrections import ysi_to_depth_below_ground
from karst_analysis.sec.io.vadose_resolver import VadoseResolver, VadoseResolution


# Map well_id ("AW6D") to the site prefix used in raw CSV filenames ("AW6_D").
def _well_id_to_filename_prefix(well_id: str) -> str:
    """Convert ``"AW6D"`` -> ``"AW6_D"`` and ``"LRS70D"`` -> ``"LRS70_D"``.

    Filenames in ``data/raw/sec/<campaign>/`` follow the convention
    ``{site}_{well_type}_YSI_{YYYYMMDD}.csv`` (with the underscore
    between site and well_type). The ``well_id`` used elsewhere in the
    code drops the underscore.
    """
    m = re.match(r"^([A-Z]+\d+)([A-Z])$", well_id)
    if not m:
        raise ValueError(
            f"Cannot parse well_id '{well_id}' into site + well_type. "
            f"Expected pattern like 'AW6D', 'LRS70D'."
        )
    site, well_type = m.group(1), m.group(2)
    return f"{site}_{well_type}"


@dataclass(frozen=True)
class RawYsiTrace:
    """One raw YSI trace (one CSV file).

    Attributes
    ----------
    well_id : str
        Well identifier, e.g. ``"AW6D"``.
    campaign : str
        Field campaign identifier, e.g. ``"2022_02"``.
    source_path : Path
        Path to the CSV that was loaded.
    date_str : str
        Date extracted from the filename (``"YYYYMMDD"``), or empty if
        the pattern did not match.
    df : pd.DataFrame
        The loaded data, standardised. Always contains ``depth_m`` and
        ``sec_uS_cm``. If a vadose thickness was provided (or could be
        looked up) a ``depth_bgl_m`` column is appended.
    vadose_thickness_m : float or None
        Vadose-zone thickness used to compute ``depth_bgl_m`` (or None
        if no shift was applied).
    vadose_resolution : VadoseResolution or None
        Diagnostic record of how ``vadose_thickness_m`` was obtained.
        Set when the trace was loaded by ``load_raw_ysi_traces_for_well``
        via the ``VadoseResolver``; carries the source ("explicit",
        "computed_from_csv", "fallback") so downstream code (plotting,
        reporting) can flag traces that used fallback values. ``None``
        if no resolver was used (e.g. when an explicit
        ``vadose_thickness_m`` was passed by the caller).
    probe : str or None
        YSI probe identifier extracted from the filename when the cast
        was made with one of multiple sondes in the same campaign
        (filenames of the form ``{site}_{type}_YSI_R_{date}.csv`` or
        ``..._Y_{date}.csv``). ``None`` for filenames without a probe
        marker (the historical convention up through 2023).
    """
    well_id: str
    campaign: str
    source_path: Path
    date_str: str
    df: pd.DataFrame = field(repr=False)
    vadose_thickness_m: Optional[float]
    vadose_resolution: Optional["VadoseResolution"] = None
    probe: Optional[str] = None


def load_raw_ysi_traces_for_well(
    well_id: str,
    campaign: str,
    *,
    project_root: Optional[Path | str] = None,
    well_type: str = "D",
    vadose_thickness_m: Optional[float] = None,
    metadata: Optional[pd.DataFrame] = None,
    add_depth_bgl: bool = True,
    column_mappings: Optional[dict[str, list[str]]] = None,
) -> List[RawYsiTrace]:
    """Load all raw YSI CSVs for a (well, campaign) pair.

    Two on-disk layouts are accepted (v12):

    * **With well_type subfolder (v10/v11 layout for 2022_02)**::

          <project_root>/data/raw/sec/<campaign>/<well_type>/
              {site}_{well_type}_YSI_{YYYYMMDD}.csv          (one or more)

    * **Without well_type subfolder (2022_08 onwards)**::

          <project_root>/data/raw/sec/<campaign>/
              {site}_{well_type}_YSI_{YYYYMMDD}.csv

    The function searches the campaign folder *recursively*, so files
    in any subfolder structure are found provided their filenames
    follow the convention. The well_type for filtering comes from
    ``well_type`` (parameter); files for other types are ignored.

    Filenames may include an optional probe marker ``R`` or ``Y``
    between the instrument tag and the date::

        {site}_{well_type}_YSI_R_{YYYYMMDD}.csv
        {site}_{well_type}_YSI_Y_{YYYYMMDD}.csv

    The probe marker is captured in ``RawYsiTrace.probe`` so downstream
    code (panel rendering) can label the trace.

    The depth-to-BGL conversion follows project convention:

        depth_bgl_m  =  depth_m  +  vadose_thickness_m

    where ``vadose_thickness_m`` is resolved by the three-level
    ``VadoseResolver`` (explicit row in wells.csv → computed from CSV
    → fallback campaign) unless the caller provides an explicit value.

    Parameters
    ----------
    well_id : str
        e.g. ``"AW6D"``. Combines ``site`` and ``well_type``.
    campaign : str
        e.g. ``"2022_02"``.
    project_root : Path or str, optional
        Defaults to ``Path.cwd()``.
    well_type : str, default ``"D"``
        Filters files by their type marker. The ``well_type`` letter
        in ``well_id`` should match this argument; mismatches raise.
    vadose_thickness_m : float, optional
        Override the resolver entirely. When given, no lookup happens.
    metadata : pd.DataFrame, optional
        Currently unused; kept for backwards compatibility with v10.
    add_depth_bgl : bool, default True
        If True, append a ``depth_bgl_m`` column when the vadose can be
        determined.
    column_mappings : dict, optional
        Custom YSI column name mappings.

    Returns
    -------
    list[RawYsiTrace]
        One element per matching CSV, sorted by ``(date_str, probe)``
        ascending. Empty list if no files match.

    Raises
    ------
    FileNotFoundError
        If the campaign folder does not exist.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    campaign_folder = root / "data" / "raw" / "sec" / campaign
    if not campaign_folder.exists():
        raise FileNotFoundError(
            f"Campaign folder not found: {campaign_folder}. "
            f"Expected layout: data/raw/sec/<campaign>/[<well_type>/]<file>.csv"
        )

    prefix = _well_id_to_filename_prefix(well_id)
    # Filename: {site}_{well_type}_YSI_(probe_)?{date}.csv  — probe is
    # optional and limited to a single uppercase letter (R or Y so far).
    pattern = re.compile(
        rf"^{re.escape(prefix)}_YSI_(?:([A-Z])_)?(\d{{8}})\.csv$"
    )

    matches: List[Path] = []
    date_strs: List[str] = []
    probes: List[Optional[str]] = []
    for f in campaign_folder.rglob("*.csv"):
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            matches.append(f)
            probes.append(m.group(1))   # None when no probe marker present
            date_strs.append(m.group(2))

    if not matches:
        return []

    # Sort by (date, probe) so identical-date sondes group deterministically.
    triples = sorted(
        zip(matches, date_strs, probes),
        key=lambda t: (t[1], t[2] or ""),
    )

    # Resolve vadose using the three-level policy unless the caller
    # provided an override. Constructing the resolver is cheap; we do
    # it once per call.
    resolver: Optional[VadoseResolver] = None
    if vadose_thickness_m is None and add_depth_bgl:
        try:
            resolver = VadoseResolver(
                metadata_csv_path=root / "data" / "metadata" / "wells.csv",
            )
        except FileNotFoundError:
            resolver = None  # depth_bgl_m simply won't be added

    traces: List[RawYsiTrace] = []
    for path, date_str, probe in triples:
        df = load_ysi_csv(path, column_mappings=column_mappings, standardise=True)

        # Per-trace vadose: the resolver may compute from THIS CSV,
        # so we need to call it once per file.
        trace_vadose: Optional[float] = vadose_thickness_m
        trace_resolution: Optional[VadoseResolution] = None
        if vadose_thickness_m is None and resolver is not None:
            try:
                trace_resolution = resolver.resolve(
                    well_id=well_id, campaign=campaign, csv_path=path,
                )
                trace_vadose = trace_resolution.thickness_m
            except KeyError:
                trace_vadose = None  # nothing worked; depth_bgl_m skipped

        if add_depth_bgl and trace_vadose is not None and "depth_m" in df.columns:
            df = df.copy()
            df["depth_bgl_m"] = ysi_to_depth_below_ground(
                df["depth_m"].values, trace_vadose,
            )

        traces.append(RawYsiTrace(
            well_id=well_id,
            campaign=campaign,
            source_path=path,
            date_str=date_str,
            df=df,
            vadose_thickness_m=trace_vadose,
            vadose_resolution=trace_resolution,
            probe=probe,
        ))

    return traces

