"""Vadose-zone thickness resolution policy for the karst_analysis project.

Background
----------
Each SEC profile is recorded in YSI's water-table-zero datum, but every
downstream technique (caliper, video log, drilling logs, ERT) lives in
ground-level-zero. To convert one to the other we need the vadose-zone
thickness for that (well, campaign).

Vadose thickness varies between campaigns: the water table moves with
seasons, recharge, tides. So in principle each (well, campaign) needs
its own value. In practice we have three sources of evidence, listed
from most to least authoritative:

    1. Explicit measurement recorded in ``data/metadata/wells.csv``.
       Either taken in the field or computed from a previous run of
       ``scripts/extract_vadose_from_raw.py``.

    2. Computed on the fly from the YSI CSV itself, when the file
       happens to include both ``Vertical Position m`` and ``Depth from
       GL (m)`` columns. This is the same operation that originally
       populated ``wells.csv`` for Feb-2022.

    3. Fallback to the value of a reference campaign — by default
       Feb-2022, the campaign that has the most carefully measured
       vadose values. Used only when (1) and (2) fail.

This module collapses the three-level lookup behind a single class so
that callers don't sprinkle fallback logic across the codebase.

The behaviour and the source of each resolution are reported in a
``VadoseResolution`` value object so that downstream code (plotting,
reporting) can flag values that came from fallback rather than from
direct measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from karst_analysis.corrections.datum import extract_vadose_from_ysi_csv


# Where wells.csv lives by default.
DEFAULT_METADATA_PATH = Path("data/metadata/wells.csv")

# Campaign whose vadose values are used as the safety net when the
# requested (well, campaign) cannot be resolved any other way.
DEFAULT_FALLBACK_CAMPAIGN = "2022_02"


VadoseSource = Literal["explicit", "computed_from_csv", "fallback"]


@dataclass(frozen=True)
class VadoseResolution:
    """Outcome of resolving the vadose-zone thickness for a (well, campaign).

    Attributes
    ----------
    thickness_m : float
        Vadose-zone thickness in metres (always positive).
    source : VadoseSource
        Which level of the policy succeeded. ``"explicit"`` means the
        value was found verbatim in ``wells.csv`` for the requested
        (well, campaign). ``"computed_from_csv"`` means it was derived
        from the YSI raw CSV's own columns. ``"fallback"`` means the
        requested (well, campaign) was not available and we used the
        reference campaign's value instead.
    well_id : str
        Well identifier (e.g. ``"AW6D"``).
    campaign : str
        Campaign identifier as requested by the caller.
    fallback_campaign : str or None
        Set only when ``source == "fallback"``: which campaign donated
        the value (``"2022_02"`` by default).
    note : str
        Human-readable diagnostic, e.g.
        ``"loaded from wells.csv row (AW6, D, 2022_02)"``.
    """
    thickness_m: float
    source: VadoseSource
    well_id: str
    campaign: str
    fallback_campaign: Optional[str]
    note: str

    @property
    def is_fallback(self) -> bool:
        """True if the value did not come from the requested (well, campaign)."""
        return self.source == "fallback"


class VadoseResolver:
    """Resolves vadose-zone thickness for any (well, campaign) using a
    three-level policy: explicit lookup → CSV computation → fallback.

    Parameters
    ----------
    metadata_csv_path : Path-like, optional
        Path to ``wells.csv``. Defaults to
        ``data/metadata/wells.csv`` relative to CWD.
    fallback_campaign : str, default ``"2022_02"``
        The reference campaign used at the bottom of the policy stack
        when the requested (well, campaign) cannot be resolved by
        steps 1 or 2.

    Raises
    ------
    FileNotFoundError
        If ``wells.csv`` does not exist at the given path.
    ValueError
        If ``wells.csv`` is missing required columns.

    Notes
    -----
    The metadata table is loaded once, at construction time. If the
    file changes on disk during a session, instantiate a new resolver.
    The class is intentionally cheap to construct.

    The metadata table is expected to have at minimum these columns:

        site, well_type, vadose_thickness_m

    The optional ``campaign`` column was introduced in v11 to support
    multiple campaigns per well. If the column is absent, every row is
    treated as belonging to the fallback campaign — this preserves
    backwards compatibility with the v10 ``wells.csv`` schema where
    the campaign was implicit (always Feb-2022).
    """

    REQUIRED_COLUMNS = ("site", "well_type", "vadose_thickness_m")

    def __init__(
        self,
        metadata_csv_path: Optional[str | Path] = None,
        fallback_campaign: str = DEFAULT_FALLBACK_CAMPAIGN,
    ) -> None:
        self._metadata_path = (
            Path(metadata_csv_path)
            if metadata_csv_path is not None
            else DEFAULT_METADATA_PATH
        )
        self._fallback_campaign = fallback_campaign
        self._table = self._load_metadata(self._metadata_path)

    # ─── Loading ────────────────────────────────────────────────────
    @classmethod
    def _load_metadata(cls, path: Path) -> pd.DataFrame:
        """Load wells.csv and validate its schema."""
        if not path.exists():
            raise FileNotFoundError(
                f"wells.csv not found at '{path}'. "
                f"Expected columns: {list(cls.REQUIRED_COLUMNS)}"
                + " plus the optional 'campaign' column."
            )

        df = pd.read_csv(path)
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"wells.csv at '{path}' is missing required columns: "
                f"{sorted(missing)}"
            )

        # Build a well_id column for lookup.
        df["well_id"] = df["site"].astype(str) + df["well_type"].astype(str)

        # If the optional 'campaign' column is absent, treat every row
        # as belonging to the (single) reference campaign. This matches
        # the v10 schema where the campaign was implicit.
        if "campaign" not in df.columns:
            df["campaign"] = DEFAULT_FALLBACK_CAMPAIGN

        return df

    @property
    def metadata(self) -> pd.DataFrame:
        """Internal lookup table (read-only view; do not mutate)."""
        return self._table

    @property
    def fallback_campaign(self) -> str:
        return self._fallback_campaign

    # ─── Resolution ─────────────────────────────────────────────────
    def resolve(
        self,
        well_id: str,
        campaign: str,
        csv_path: Optional[Path | str] = None,
    ) -> VadoseResolution:
        """Resolve the vadose-zone thickness for one (well, campaign).

        The policy is tried strictly in order; the first hit wins.

        Parameters
        ----------
        well_id : str
            e.g. ``"AW6D"``.
        campaign : str
            e.g. ``"2023_02"``.
        csv_path : Path-like, optional
            Path to the YSI raw CSV for this (well, campaign). Used
            ONLY by step 2 of the policy (extraction from CSV columns).
            If not given, step 2 is skipped silently.

        Returns
        -------
        VadoseResolution
            Always returns a result. If no level of the policy fired,
            raises ``KeyError`` (see below) — never returns None.

        Raises
        ------
        KeyError
            If neither the requested (well, campaign) nor the well in
            the fallback campaign can be found in ``wells.csv``, AND
            the CSV computation either failed or was not attempted.
        """
        # Level 1 — explicit lookup
        hit = self._lookup_explicit(well_id, campaign)
        if hit is not None:
            return VadoseResolution(
                thickness_m=hit,
                source="explicit",
                well_id=well_id,
                campaign=campaign,
                fallback_campaign=None,
                note=f"loaded from wells.csv row ({well_id}, {campaign})",
            )

        # Level 2 — computed from the YSI CSV directly
        if csv_path is not None:
            csv_path = Path(csv_path)
            if csv_path.exists():
                offset, status = extract_vadose_from_ysi_csv(csv_path)
                if offset is not None and status == "ok":
                    return VadoseResolution(
                        thickness_m=offset,
                        source="computed_from_csv",
                        well_id=well_id,
                        campaign=campaign,
                        fallback_campaign=None,
                        note=(
                            f"computed from {csv_path.name} "
                            f"(GL minus Vertical Position; status={status})"
                        ),
                    )

        # Level 3 — fallback to the reference campaign
        fallback_hit = self._lookup_explicit(well_id, self._fallback_campaign)
        if fallback_hit is not None:
            return VadoseResolution(
                thickness_m=fallback_hit,
                source="fallback",
                well_id=well_id,
                campaign=campaign,
                fallback_campaign=self._fallback_campaign,
                note=(
                    f"requested ({well_id}, {campaign}) not found; "
                    f"using {well_id} value from fallback campaign "
                    f"'{self._fallback_campaign}'"
                ),
            )

        # Nothing worked.
        raise KeyError(
            f"Cannot resolve vadose for ({well_id}, {campaign}): "
            f"no explicit row in wells.csv, no usable CSV provided, "
            f"and no fallback row for '{well_id}' in campaign "
            f"'{self._fallback_campaign}'."
        )

    def _lookup_explicit(self, well_id: str, campaign: str) -> Optional[float]:
        """Return the explicit value for (well_id, campaign), or None."""
        mask = (self._table["well_id"] == well_id) & (
            self._table["campaign"] == campaign
        )
        rows = self._table[mask]
        if rows.empty:
            return None
        # If there happen to be multiple rows for the same (well, campaign)
        # we keep the first; this should not happen on a healthy wells.csv.
        value = rows.iloc[0]["vadose_thickness_m"]
        if pd.isna(value):
            return None
        return float(value)
