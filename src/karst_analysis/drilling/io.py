"""Loader for the Ardaman lithology / in-situ-conductivity CSV.

The Ardaman & Associates 2009 report describes core borings B-5 and B-6
(modern names AW5O and AW6O respectively). Each lithology row marks the
TOP of a unit; the bottom is implied by the top of the next lithology
unit. In-situ conductivity rows are point measurements (top == bottom).

The csv has comment lines starting with ``#`` (skipped) and the
following columns:

    well, depth_ft, depth_m, kind, text

where ``kind`` ∈ {``"lithology"``, ``"conductivity_in_situ"``}.

Migration history
-----------------
v5.1: extracted from ``caliper_videolog_panel.py``. No algorithmic
changes. The interval extension logic for lithology rows is preserved
verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_ARDAMAN_CSV = Path("data/raw/drilling/ardaman_lithology.csv")


def load_ardaman(
    csv_path: Optional[str | Path] = None,
    *,
    well: str,
) -> pd.DataFrame:
    """Load Ardaman lithology + in-situ-conductivity entries for one well.

    Lithology rows mark the top of a unit; the loader extends the row
    to span until the top of the next lithology unit. Conductivity rows
    are kept as point measurements (top == bottom).

    Parameters
    ----------
    csv_path : str or Path, optional
        Defaults to :data:`DEFAULT_ARDAMAN_CSV`.
    well : str
        Modern well name (``"AW5O"`` or ``"AW6O"``). Returns an empty
        DataFrame if the well is absent from the file.

    Returns
    -------
    pd.DataFrame
        Columns: ``depth_top_m``, ``depth_bot_m`` (NaN for the deepest
        lithology unit), ``kind``, ``text``, ``depth_top_bgl_m``,
        ``depth_bot_bgl_m``, ``depth_centre_bgl_m``. Sorted by
        ``depth_centre_bgl_m`` ascending.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_ARDAMAN_CSV
    if not path.exists():
        raise FileNotFoundError(f"Ardaman CSV not found: {path}")

    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    sub = df[df["well"] == well].sort_values("depth_m").reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()

    # Lithology rows describe the TOP of a unit. The bottom is the top
    # of the next lithology unit. The deepest lithology unit has no
    # known bottom (NaN). Conductivity rows are point measurements
    # (bot == top).
    is_lith = sub["kind"] == "lithology"
    lith_idx = sub.index[is_lith].tolist()
    bot = sub["depth_m"].to_numpy(dtype=float).copy()
    for k, idx in enumerate(lith_idx):
        if k + 1 < len(lith_idx):
            bot[idx] = float(sub.loc[lith_idx[k + 1], "depth_m"])
        else:
            bot[idx] = np.nan
    for idx in sub.index[~is_lith]:
        bot[idx] = float(sub.loc[idx, "depth_m"])

    out = pd.DataFrame({
        "depth_top_m": sub["depth_m"].to_numpy(dtype=float),
        "depth_bot_m": bot,
        "kind":        sub["kind"].to_numpy(dtype=object),
        "text":        sub["text"].to_numpy(dtype=object),
    })
    # BGL-positive depths. Columns named ``depth_*_bgl_m`` so downstream
    # consumers know they share datum with SEC / caliper / videolog.
    # Reserve "elevation" for the absolute (above sea level) case.
    out["depth_top_bgl_m"] = out["depth_top_m"]
    out["depth_bot_bgl_m"] = out["depth_bot_m"]
    out["depth_centre_bgl_m"] = np.where(
        np.isfinite(out["depth_bot_bgl_m"]),
        0.5 * (out["depth_top_bgl_m"] + out["depth_bot_bgl_m"]),
        out["depth_top_bgl_m"],
    )
    return (out.sort_values("depth_centre_bgl_m", ascending=True)
               .reset_index(drop=True))
