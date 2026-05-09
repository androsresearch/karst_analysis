"""Loaders and data structures for ERT 1D resistivity profiles.

ERT 1D files in this project are columns extracted at fixed x positions
from 2D inverted resistivity sections (produced upstream in ResIPy from
joint dipole-dipole + Wenner data, then post-processed). The same x
position can have several **variants** — different post-processing or
different inversion parameters — exported as separate CSVs.

File schema (input)
-------------------
    x_requested, x_extracted, depth, resist, resistlog10
    -- depth is NEGATIVE (zero at ground level, depth grows downward
       in magnitude). The loader flips the sign and adds depth_bgl_m
       (positive down) to match the project convention used by SEC and
       caliper.

File location convention
------------------------
    <project_root>/data/raw/ert/<transect>/1D/<variant>_x_<x>.csv

    where:
      - <transect> is e.g. "T16"
      - <variant> can contain underscores ("error_weighted",
        "tier3_robust_blocky", ...) -- the loader uses the LAST "_x_"
        token as the split point.
      - <x> matches x_requested, integer or decimal.

Well -> transect mapping
------------------------
A separate CSV ``data/metadata/ert_wells.csv`` maps wells to (transect,
x) tuples. One well can map to multiple (transect, x) entries; one
(transect, x) can host multiple wells if the inversion line passes
near several boreholes. The loader does NOT care about this mapping;
the ``ErtWellMap`` class does.

Design choices
--------------
- ``ErtTrace1D`` is frozen and carries a DataFrame. The DataFrame's
  schema is part of the contract (see ``REQUIRED_COLUMNS`` below).
- ``load_ert_1d_traces`` raises ``FileNotFoundError`` for missing
  transects (rather than returning an empty list), to disambiguate
  "transect does not exist" from "transect is empty".
- ``x_filter`` raises ``ValueError`` listing the available x values
  rather than returning an empty list. The user wants to know what
  IS available, not silently get nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ── Public schema contracts ─────────────────────────────────────────
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "x_requested", "x_extracted", "depth", "resist", "resistlog10",
)
REQUIRED_TRACE_COLUMNS: tuple[str, ...] = (
    "depth_bgl_m", "resist_ohm_m", "resistlog10",
)

# Filename pattern: "<variant>_x_<number>.csv"
# variant captures everything before the LAST "_x_"; x captures the
# trailing numeric token (int or decimal).
_FNAME_RE = re.compile(r"^(?P<variant>.+)_x_(?P<x>-?\d+(?:\.\d+)?)\.csv$")


@dataclass(frozen=True)
class ErtTrace1D:
    """A single 1D ERT resistivity profile (one variant, one x position).

    Attributes
    ----------
    transect : str
        Transect identifier, e.g. "T16".
    x_requested : float
        The x position requested for extraction (from the filename).
    x_extracted : float
        The closest mesh column actually returned by the inversion
        (read from the CSV; usually slightly different from x_requested
        because of mesh discretisation).
    variant : str
        Post-processing or inversion variant label, parsed from the
        filename, e.g. "viz_sharp", "tier3_robust_blocky",
        "error_weighted".
    source_path : Path
        Absolute path to the source CSV.
    df : pd.DataFrame
        Tidy DataFrame, sorted ascending by ``depth_bgl_m``. Required
        columns:
          - ``depth_bgl_m``  : depth, positive down (project convention;
                                obtained by flipping the sign of the
                                ``depth`` column in the source CSV)
          - ``resist_ohm_m`` : resistivity (linear)
          - ``resistlog10``  : log10 of resistivity
    """

    transect: str
    x_requested: float
    x_extracted: float
    variant: str
    source_path: Path
    df: pd.DataFrame = field(repr=False)


@dataclass(frozen=True)
class ErtWellAssoc:
    """Association between a well and a (transect, x) point on an ERT line.

    Attributes
    ----------
    well_id : str
        e.g. "LRS70D".
    transect : str
        e.g. "T16".
    x : float
        The requested x position on the transect for the 1D extraction.
    notes : str
        Free-form notes from the metadata table; empty string if none.
    """

    well_id: str
    transect: str
    x: float
    notes: str = ""


# ════════════════════════════════════════════════════════════════════
#  Filename parsing
# ════════════════════════════════════════════════════════════════════
def parse_ert_filename(filename: str) -> tuple[str, float]:
    """Parse "<variant>_x_<x>.csv" into (variant, x_requested).

    Examples
    --------
    >>> parse_ert_filename("viz_sharp_x_160.csv")
    ('viz_sharp', 160.0)
    >>> parse_ert_filename("tier3_robust_blocky_x_160.csv")
    ('tier3_robust_blocky', 160.0)
    >>> parse_ert_filename("error_weighted_x_172.5.csv")
    ('error_weighted', 172.5)
    """
    m = _FNAME_RE.match(Path(filename).name)
    if m is None:
        raise ValueError(
            f"Cannot parse ERT filename: {filename!r}. "
            f"Expected pattern '<variant>_x_<x>.csv'."
        )
    return m.group("variant"), float(m.group("x"))


# ════════════════════════════════════════════════════════════════════
#  Single-file loader
# ════════════════════════════════════════════════════════════════════
def load_ert_1d_csv(
    filepath: str | Path, *, transect: Optional[str] = None,
) -> ErtTrace1D:
    """Load a single ERT 1D CSV into an ``ErtTrace1D``.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    transect : str, optional
        Transect identifier (e.g. "T16"). If not provided, this is
        inferred from the parent directory three levels up
        (``.../<transect>/1D/<file>.csv``). Pass it explicitly for
        files outside the canonical layout.

    Returns
    -------
    ErtTrace1D

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the filename cannot be parsed or required columns are
        missing.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ERT CSV not found: {path}")

    variant, x_requested = parse_ert_filename(path.name)

    if transect is None:
        # Expected layout: .../<transect>/1D/<file>.csv
        # parents[0] = "1D", parents[1] = "<transect>"
        if len(path.parents) < 2 or path.parents[0].name != "1D":
            raise ValueError(
                f"Cannot infer transect from {path}. "
                f"Expected path to end with '/<transect>/1D/<file>.csv', "
                f"or pass `transect=` explicitly."
            )
        transect = path.parents[1].name

    df_raw = pd.read_csv(path)

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"ERT CSV {path.name} missing required columns: {missing}. "
            f"Found: {list(df_raw.columns)}"
        )

    x_extracted = float(df_raw["x_extracted"].iloc[0])
    if not np.allclose(df_raw["x_extracted"].to_numpy(),
                        x_extracted, rtol=0, atol=1e-9):
        # This would mean the file mixes columns at different x — which
        # is not a 1D profile any more.
        raise ValueError(
            f"ERT CSV {path.name} has non-constant x_extracted; "
            f"expected a single 1D column."
        )

    # Build the trace DataFrame: flip depth sign, sort by depth.
    df = pd.DataFrame({
        "depth_bgl_m": -df_raw["depth"].to_numpy(),
        "resist_ohm_m": df_raw["resist"].to_numpy(),
        "resistlog10": df_raw["resistlog10"].to_numpy(),
    })
    df = df.sort_values("depth_bgl_m", kind="stable").reset_index(drop=True)

    return ErtTrace1D(
        transect=transect,
        x_requested=x_requested,
        x_extracted=x_extracted,
        variant=variant,
        source_path=path,
        df=df,
    )


# ════════════════════════════════════════════════════════════════════
#  Per-transect loader
# ════════════════════════════════════════════════════════════════════
def load_ert_1d_traces(
    transect: str,
    *,
    project_root: Optional[str | Path] = None,
    x_filter: Optional[float] = None,
    variant_filter: Optional[Sequence[str]] = None,
) -> list[ErtTrace1D]:
    """Load all ERT 1D traces from a given transect.

    Parameters
    ----------
    transect : str
        Transect identifier, e.g. "T16".
    project_root : str or Path, optional
        Project root directory. Defaults to the current working
        directory. Files are read from
        ``<project_root>/data/raw/ert/<transect>/1D/``.
    x_filter : float, optional
        If given, return only traces whose ``x_requested`` matches
        this value (atol=1e-6). Raises ``ValueError`` if no trace
        matches, listing the available x values for the user to pick.
    variant_filter : sequence of str, optional
        If given, return only traces whose ``variant`` is in the list.

    Returns
    -------
    list[ErtTrace1D]
        Sorted by (x_requested, variant).

    Raises
    ------
    FileNotFoundError
        If the transect directory does not exist.
    ValueError
        If ``x_filter`` is given but no trace matches.
    """
    if project_root is None:
        project_root = Path.cwd()
    transect_dir = (
        Path(project_root) / "data" / "raw" / "ert" / transect / "1D"
    )
    if not transect_dir.is_dir():
        raise FileNotFoundError(
            f"ERT transect directory not found: {transect_dir}"
        )

    traces: list[ErtTrace1D] = []
    for csv_path in sorted(transect_dir.glob("*.csv")):
        try:
            traces.append(load_ert_1d_csv(csv_path, transect=transect))
        except ValueError as exc:
            # Re-raise with file context so the user knows which file
            # caused the failure.
            raise ValueError(f"While loading {csv_path.name}: {exc}") from exc

    # ── x_filter: exact match with tolerance, helpful error message ──
    if x_filter is not None:
        kept = [t for t in traces
                if np.isclose(t.x_requested, x_filter, atol=1e-6)]
        if not kept:
            available = sorted({t.x_requested for t in traces})
            raise ValueError(
                f"No ERT 1D trace found at x_requested={x_filter} on "
                f"transect {transect!r}. Available x values: "
                f"{available}"
            )
        traces = kept

    # ── variant_filter: subset by variant label ──────────────────────
    if variant_filter is not None:
        wanted = set(variant_filter)
        traces = [t for t in traces if t.variant in wanted]

    traces.sort(key=lambda t: (t.x_requested, t.variant))
    return traces


# ════════════════════════════════════════════════════════════════════
#  Well -> (transect, x) mapping
# ════════════════════════════════════════════════════════════════════
class ErtWellMap:
    """In-memory map between wells and (transect, x) ERT 1D points.

    The underlying CSV must have these columns:

        well_id,transect,x,notes

    One row per (well_id, transect, x) association. A well may have
    several rows (e.g. if it sits near more than one transect or if
    multiple x positions on the same transect were extracted for the
    same well).
    """

    REQUIRED_COLUMNS: tuple[str, ...] = ("well_id", "transect", "x", "notes")

    def __init__(self, associations: Sequence[ErtWellAssoc]) -> None:
        self._assocs: tuple[ErtWellAssoc, ...] = tuple(associations)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ErtWellMap":
        """Load the map from ``data/metadata/ert_wells.csv`` (or any path).

        Notes column may be absent; if so it's filled with empty strings.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ert_wells.csv not found: {path}")

        df = pd.read_csv(path, dtype={"well_id": str, "transect": str})
        # Tolerate missing 'notes' column.
        if "notes" not in df.columns:
            df["notes"] = ""
        df["notes"] = df["notes"].fillna("")

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"ert_wells.csv missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        assocs = [
            ErtWellAssoc(
                well_id=str(row["well_id"]).strip(),
                transect=str(row["transect"]).strip(),
                x=float(row["x"]),
                notes=str(row["notes"]),
            )
            for _, row in df.iterrows()
        ]
        return cls(assocs)

    @property
    def associations(self) -> tuple[ErtWellAssoc, ...]:
        return self._assocs

    def transects_for_well(self, well_id: str) -> list[ErtWellAssoc]:
        """All (transect, x) entries associated with a given well_id."""
        return [a for a in self._assocs if a.well_id == well_id]

    def wells_for_transect(self, transect: str) -> list[ErtWellAssoc]:
        """All (well_id, x) entries associated with a given transect."""
        return [a for a in self._assocs if a.transect == transect]

    def __len__(self) -> int:
        return len(self._assocs)

    def __repr__(self) -> str:
        return (
            f"ErtWellMap(n={len(self._assocs)} associations, "
            f"wells={sorted({a.well_id for a in self._assocs})})"
        )
