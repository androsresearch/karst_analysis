"""Auto-populate ``data/metadata/wells.csv`` from raw YSI files (v11).

YSI exports often include a ``Depth from GL (m)`` column already
referenced to ground level. When present, the per-well-per-campaign
vadose-zone thickness is just the offset between that column and
``Vertical Position m``:

    vadose_thickness_m  =  Depth_from_GL  -  Vertical_Position

(at any common row — the offset is constant within a single profile).

This script walks all CSV files under ``data/raw/sec/<campaign>/<type>/``,
extracts the offset where ``Depth from GL`` is available, and writes a
fresh ``wells.csv`` ready to use. The campaign for each row comes from
the parent folder name. Wells whose raw file is missing the column
are reported and need a manual value.

Existing entries in ``wells.csv`` are preserved when a (well, campaign)
pair has no extractable raw — so manual values you entered are not
overwritten by NaN.

The single-file extraction is performed by
``karst_analysis.corrections.extract_vadose_from_ysi_csv``; this module
contains only the CLI / batch-walking logic.

v11 schema
----------
The output ``wells.csv`` has one row per (site, well_type, campaign).
Columns:

    site, well_type, campaign, vadose_thickness_m,
    reference_date, source, notes

Rows from the existing wells.csv whose (site, well_type, campaign) is
NOT touched by this run are kept verbatim — never silently dropped.

Usage
-----
    uv run python scripts/extract_vadose_from_raw.py
    uv run python scripts/extract_vadose_from_raw.py --raw-root data/raw/sec
    uv run python scripts/extract_vadose_from_raw.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from karst_analysis.corrections import extract_vadose_from_ysi_csv
from karst_analysis.io import parse_well_filename


# Required output schema (column order is intentional).
OUTPUT_COLUMNS = [
    "site",
    "well_type",
    "campaign",
    "vadose_thickness_m",
    "reference_date",
    "source",
    "notes",
]


def _campaign_from_path(csv_path: Path, raw_root: Path) -> str | None:
    """Extract the campaign identifier from the CSV's parent folder.

    Expected layout: ``<raw_root>/<campaign>/<well_type>/<file>.csv``.
    Returns ``None`` if the layout doesn't match.
    """
    try:
        rel = csv_path.relative_to(raw_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:  # need at least campaign / well_type / file
        return None
    return parts[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", default="data/raw/sec",
        help="Root folder to walk for CSV files (default: data/raw/sec).",
    )
    parser.add_argument(
        "--out", default="data/metadata/wells.csv",
        help="Output CSV path (default: data/metadata/wells.csv).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without modifying the file.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_path = Path(args.out)

    if not raw_root.exists():
        print(f"ERROR: raw root does not exist: {raw_root}", file=sys.stderr)
        return 1

    # Load any existing rows so we can preserve manual values for
    # (well, campaign) pairs that this run does not touch.
    existing_rows: dict[tuple[str, str, str], dict] = {}
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        # Backwards compat: an old wells.csv without the 'campaign' column
        # is treated as belonging to the reference campaign 2022_02. The
        # row is then effectively migrated to the v11 schema.
        if "campaign" not in df_existing.columns:
            print(
                "  NOTE: existing wells.csv has no 'campaign' column — "
                "treating every row as 2022_02 and migrating to v11 schema."
            )
            df_existing["campaign"] = "2022_02"
        for _, row in df_existing.iterrows():
            key = (str(row["site"]), str(row["well_type"]),
                   str(row["campaign"]))
            existing_rows[key] = row.to_dict()

    csvs = sorted(raw_root.rglob("*.csv"))
    print(f"Walking {raw_root} — found {len(csvs)} CSV files\n")

    # Keyed by (site, well_type, campaign) — one row per pair.
    rows: dict[tuple[str, str, str], dict] = {}
    issues: list[tuple[str, str]] = []

    for csv_path in csvs:
        try:
            info = parse_well_filename(csv_path)
        except ValueError as e:
            issues.append((csv_path.name, f"unparseable filename: {e}"))
            continue

        campaign = _campaign_from_path(csv_path, raw_root)
        if campaign is None:
            issues.append(
                (csv_path.name,
                 f"could not infer campaign from path "
                 f"{csv_path} (expected <raw_root>/<campaign>/<type>/<file>)")
            )
            continue

        offset, status = extract_vadose_from_ysi_csv(csv_path)
        well_id = info.well_id
        key = (info.site, info.well_type, campaign)

        if status == "ok" or status.startswith("inconsistent"):
            # If multiple CSVs exist for the same (well, campaign) pair
            # (e.g. several casts on different days), keep the most
            # recent one.
            existing_for_key = rows.get(key)
            if existing_for_key is None or info.date > existing_for_key["_date_raw"]:
                rows[key] = {
                    "site": info.site,
                    "well_type": info.well_type,
                    "campaign": campaign,
                    "vadose_thickness_m":
                        round(offset, 4) if offset is not None else np.nan,
                    "reference_date":
                        f"{info.date[:4]}-{info.date[4:6]}-{info.date[6:]}",
                    "_date_raw": info.date,  # internal, dropped before write
                    "source": "extracted from Depth from GL (m)",
                    "notes": "" if status == "ok" else status,
                }
            print(
                f"  ✓ {csv_path.name:<40} {well_id:<8} {campaign:<8} "
                f"offset={offset:.4f} m  [{status}]"
            )
        else:
            # Column missing — keep an existing row for this exact
            # (site, well_type, campaign) if present.
            if key in existing_rows:
                rows.setdefault(key, {
                    "site": info.site,
                    "well_type": info.well_type,
                    "campaign": campaign,
                    "vadose_thickness_m":
                        existing_rows[key].get("vadose_thickness_m"),
                    "reference_date":
                        existing_rows[key].get("reference_date"),
                    "_date_raw": "",
                    "source": existing_rows[key].get("source", "manual"),
                    "notes": (
                        f"preserved from existing wells.csv ({status})"
                    ),
                })
                print(
                    f"  • {csv_path.name:<40} {well_id:<8} {campaign:<8} "
                    f"no Depth-from-GL — keeping existing manual value"
                )
            else:
                issues.append((
                    csv_path.name,
                    f"{status} for ({well_id}, {campaign}) and no manual "
                    f"value in wells.csv yet"
                ))
                print(
                    f"  ✗ {csv_path.name:<40} {well_id:<8} {campaign:<8} "
                    f"no Depth-from-GL and no manual value"
                )

    # Add any existing rows that were NOT touched by this run, so manual
    # entries for campaigns whose raw files we don't see don't disappear.
    for key, row in existing_rows.items():
        if key not in rows:
            preserved = dict(row)
            preserved["_date_raw"] = ""
            rows[key] = preserved

    if not rows:
        print("\nNo wells extracted. Nothing to write.")
        return 1

    # Build output table
    out_df = pd.DataFrame.from_records(list(rows.values()))
    out_df = out_df.drop(columns=["_date_raw"], errors="ignore")
    # Ensure all required columns exist (older preserved rows may lack some).
    for col in OUTPUT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[OUTPUT_COLUMNS]
    out_df = out_df.sort_values(
        ["site", "well_type", "campaign"]
    ).reset_index(drop=True)

    print()
    print("Final wells.csv contents:")
    print(out_df.to_string(index=False))

    if issues:
        print(f"\n{len(issues)} issue(s) found — these need manual entries:")
        for name, msg in issues:
            print(f"  - {name}: {msg}")

    if args.dry_run:
        print("\n[dry-run] Not writing to disk.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
