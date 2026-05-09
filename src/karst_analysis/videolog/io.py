"""Loader for the video-log xlsx file.

Each priority well has its own sheet in
``Priority_Ewan_video_logs_v2.xlsx``; the sheet name does NOT always
match the well_id of the calipered well (e.g. for AW5D the video sheet
is called ``"AW5"``). The mapping is owned by the convergence panel
config.

Migration history
-----------------
v5.1: extracted from ``caliper_videolog_panel.py``. No algorithmic
changes; the cleaned DataFrame is byte-identical to what the original
produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.videolog.parsing import apply_typo_fixes, parse_depth_token


DEFAULT_VIDEOLOG_XLSX = Path("data/raw/videolog/Priority_Ewan_video_logs_v2.xlsx")


def load_video_notes(
    xlsx_path: Optional[str | Path] = None,
    *,
    sheet: str,
) -> pd.DataFrame:
    """Load and clean the video-log notes for one well sheet.

    The xlsx has a header row at index 1 (row 2 in Excel). Each row has
    a depth cell and a notes cell. Some rows have notes but no depth —
    these are interpreted as continuations of the previous row's note
    and appended with a "; " separator.

    Parameters
    ----------
    xlsx_path : str or Path, optional
        Path to the workbook. Defaults to
        :data:`DEFAULT_VIDEOLOG_XLSX`.
    sheet : str
        Sheet name to read. Use the mapping in the convergence panel's
        ``WellConfig`` rather than guessing from a well_id.

    Returns
    -------
    pd.DataFrame
        Columns: ``depth_top_m``, ``depth_bot_m``, ``note``,
        ``depth_top_bgl_m`` (= ``depth_top_m``, kept for naming clarity
        in cross-package contexts), ``depth_bot_bgl_m``,
        ``depth_centre_bgl_m``. Sorted by ``depth_centre_bgl_m``
        ascending (surface first). Empty if the sheet has no usable rows.
    """
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_VIDEOLOG_XLSX
    if not path.exists():
        raise FileNotFoundError(f"Video-log xlsx not found: {path}")

    raw = pd.read_excel(path, sheet_name=sheet, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    depth_col = next((c for c in raw.columns if "Depth" in c), None)
    notes_col = next((c for c in raw.columns
                      if c.lower().startswith("notes")), None)
    if depth_col is None or notes_col is None:
        raise KeyError(
            f"Sheet '{sheet}' is missing a Depth or Notes column. "
            f"Columns found: {raw.columns.tolist()}"
        )

    rows: list[dict] = []
    for _, r in raw.iterrows():
        depth_token = r[depth_col]
        note = r[notes_col]
        if pd.isna(note) or not str(note).strip():
            continue
        z_top, z_bot = parse_depth_token(depth_token)
        if z_top is None:
            # Continuation row: append to the previous entry's note.
            if rows:
                rows[-1]["note"] = (
                    rows[-1]["note"].rstrip(". ").rstrip()
                    + "; " + str(note).strip()
                )
            continue
        rows.append(dict(
            depth_top_m=z_top,
            depth_bot_m=z_bot,
            note=str(note).strip(),
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["note"] = df["note"].map(apply_typo_fixes)
    # BGL-positive depths. The columns are kept as ``depth_*_bgl_m`` so that
    # downstream consumers know they're in the same datum as the SEC and
    # caliper sub-packages. The word "elevation" is reserved for absolute
    # elevation (m above sea level) once differential GPS is available.
    df["depth_top_bgl_m"]    = df["depth_top_m"]
    df["depth_bot_bgl_m"]    = df["depth_bot_m"]
    df["depth_centre_bgl_m"] = 0.5 * (df["depth_top_bgl_m"] + df["depth_bot_bgl_m"])
    return (df.sort_values("depth_centre_bgl_m", ascending=True)
              .reset_index(drop=True))
