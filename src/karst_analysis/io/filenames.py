"""Parse well metadata from CSV filenames.

Filename conventions encountered in the project:

    AW6_D_YSI_20220215.csv          → site=AW6, well_type=D, instrument=YSI, date=20220215
    AW6D_YSI_20220215.csv           → site=AW6, well_type=D, instrument=YSI, date=20220215
    LRS70_D_YSI_20220131.csv        → site=LRS70
    LRS70_D_YSI_20220131_processed  → same, with optional suffix

This module is deliberately permissive: it accepts both the underscored
form (``AW6_D``) and the concatenated form (``AW6D``) for backwards
compatibility with the existing data folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Site codes seen in the project. Extend here when new sites appear.
SITE_PATTERNS = [
    r"AW\d+",      # AW1, AW2, ..., AW7
    r"BW\d+",      # BW3, BW4, ...
    r"LRS\d+",     # LRS69, LRS70, ...
]
WELL_TYPES = ["D", "O", "S"]      # Deep, Old, Shallow
INSTRUMENTS = ["YSI", "TOM"]


@dataclass(frozen=True)
class WellFilenameInfo:
    """Parsed components of a well CSV filename."""
    site: str
    well_type: str
    instrument: str
    date: str           # YYYYMMDD
    suffix: str | None  # e.g. "processed", or None
    well_id: str        # site + well_type, e.g. "AW6D"

    def base_id(self) -> str:
        """Canonical short identifier for downstream use."""
        return f"{self.well_id}_{self.date}"


def parse_well_filename(filename: str | Path) -> WellFilenameInfo:
    """Decode a well CSV filename into its components.

    Parameters
    ----------
    filename : str or Path
        Either just the filename (``"AW6D_YSI_20220215.csv"``) or a full path.

    Returns
    -------
    WellFilenameInfo

    Raises
    ------
    ValueError
        If the filename cannot be parsed against the known patterns.
    """
    name = Path(filename).stem  # strip directory and extension

    # Try matching site code first, allowing optional underscore + well_type
    site_pattern = "|".join(SITE_PATTERNS)
    well_pattern = "|".join(WELL_TYPES)
    instr_pattern = "|".join(INSTRUMENTS)

    # Pattern A: SITE_TYPE_INSTR_DATE  (e.g. "AW6_D_YSI_20220215")
    pattern_a = re.compile(
        rf"^({site_pattern})_({well_pattern})_({instr_pattern})_(\d{{8}})(?:_(.+))?$"
    )
    # Pattern B: SITE+TYPE_INSTR_DATE  (e.g. "AW6D_YSI_20220215")
    pattern_b = re.compile(
        rf"^({site_pattern})({well_pattern})_({instr_pattern})_(\d{{8}})(?:_(.+))?$"
    )

    for pattern in (pattern_a, pattern_b):
        match = pattern.match(name)
        if match:
            site, well_type, instrument, date, suffix = match.groups()
            return WellFilenameInfo(
                site=site,
                well_type=well_type,
                instrument=instrument,
                date=date,
                suffix=suffix,
                well_id=f"{site}{well_type}",
            )

    raise ValueError(
        f"Could not parse filename '{name}'. "
        f"Expected formats: SITE_TYPE_INSTR_DATE or SITETYPE_INSTR_DATE, "
        f"with SITE in {SITE_PATTERNS}, TYPE in {WELL_TYPES}, INSTR in {INSTRUMENTS}."
    )
