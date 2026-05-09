"""Regenerate breakpoint comparison figures from existing JSONs.

For every (well, campaign) pair that has both a SavGol and a LOWESS
breakpoints JSON in ``data/breakpoints/<campaign>/``, this script
generates one figure per N (1..max_n_breakpoints) showing:

    [savgol panel | lowess panel]   for that N

Markers are placed at the (depth, sec) coordinates given by the fitted
piecewise-linear model — NOT by interpolation onto the smoothed curve.
Both axes (depth and SEC) are therefore mathematically consistent with
the fit.

The script DOES NOT re-fit anything. It only reads JSONs already on
disk. Re-run it after changing visualisation code (e.g. when
breakpoints_overlay.py is updated) to refresh all figures cheaply.

Inputs (auto-discovered)
-------------------------
    data/breakpoints/<campaign>/{well_id}_{date}__bp-savgol-*.json
    data/breakpoints/<campaign>/{well_id}_{date}__bp-lowess-*.json
    data/processed/sec/<campaign>/savgol/{well_id}_{date}__*.csv
    data/processed/sec/<campaign>/lowess/{well_id}_{date}__*.csv
    data/raw/sec/<campaign>/<type>/<original raw filename>.csv

Outputs
-------
    results/figures/breakpoints/<campaign>/{well_id}_compare_N1to{n_max}/
        {well_id}_{date}_N{nn}.png

Usage
-----
    uv run python scripts/regenerate_breakpoint_figures.py
    uv run python scripts/regenerate_breakpoint_figures.py --campaign 2022_02
    uv run python scripts/regenerate_breakpoint_figures.py --campaign 2022_02 --only LRS70D AW6D
    uv run python scripts/regenerate_breakpoint_figures.py --campaign 2022_02 --well-type D
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from karst_analysis.io import parse_well_filename
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.viz import (
    load_bic_json,
    plot_breakpoints_compare_methods,
)


# ─────────────────────────────────────────────────────────────────────────
#  Discovery + matching helpers
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class WellRunBundle:
    """All paths needed to render figures for one (well_id, date) pair."""
    well_id: str
    date: str
    raw_csv: Path
    savgol_csv: Path
    lowess_csv: Path
    savgol_json: Path
    lowess_json: Path


@dataclass
class SkippedRun:
    """Records why a (well_id, date) pair could not be rendered."""
    well_id: str
    date: str
    reason: str
    missing: list[str] = field(default_factory=list)


def _parse_well_id(well_id: str) -> tuple[str, str]:
    """Split a well_id like 'AW5D' into ('AW5', 'D'). Raises if malformed."""
    m = re.match(r"^(.+?)([DSO])$", well_id)
    if m is None:
        raise ValueError(f"Cannot split well_id '{well_id}' into site+type")
    return m.group(1), m.group(2)


def _find_raw_csv(raw_dir: Path, well_id: str, date: str) -> Optional[Path]:
    """Find the raw YSI file for a given well_id+date.

    Tries both filename conventions:
      - SITE_TYPE_INSTR_DATE.csv  (e.g. AW5_D_YSI_20220213.csv)
      - SITETYPE_INSTR_DATE.csv   (e.g. AW5D_YSI_20220213.csv)
    """
    site, well_type = _parse_well_id(well_id)
    candidates = list(raw_dir.glob(f"{site}_{well_type}_*_{date}.csv"))
    if not candidates:
        candidates = list(raw_dir.glob(f"{site}{well_type}_*_{date}.csv"))
    if not candidates:
        return None
    return candidates[0]


def _find_processed_csv(proc_dir: Path, well_id: str, date: str) -> Optional[Path]:
    """Find a processed CSV with the given well_id+date prefix."""
    candidates = sorted(proc_dir.glob(f"{well_id}_{date}__*.csv"))
    return candidates[-1] if candidates else None


def _discover_bundles(
    *,
    bp_dir: Path,
    raw_dir: Path,
    sav_dir: Path,
    low_dir: Path,
    only_wells: Optional[set[str]] = None,
    only_well_type: Optional[str] = None,
) -> tuple[list[WellRunBundle], list[SkippedRun]]:
    """Walk the breakpoints folder and assemble bundles for renderable runs."""
    bundles: list[WellRunBundle] = []
    skipped: list[SkippedRun] = []

    sav_jsons = sorted(bp_dir.glob("*__bp-savgol-*.json"))

    for sav_json in sav_jsons:
        # Filename: {well_id}_{date}__bp-savgol-...
        stem = sav_json.stem
        try:
            well_date_part, _method_sig = stem.split("__", 1)
            well_id, date = well_date_part.rsplit("_", 1)
        except ValueError:
            skipped.append(SkippedRun(
                well_id="?", date="?",
                reason=f"unparseable JSON filename: {sav_json.name}",
            ))
            continue

        # Optional well_type / well filter
        try:
            _site, well_type = _parse_well_id(well_id)
        except ValueError as e:
            skipped.append(SkippedRun(
                well_id=well_id, date=date,
                reason=str(e),
            ))
            continue

        if only_well_type and well_type != only_well_type:
            continue
        if only_wells and well_id not in only_wells:
            continue

        # Companion lowess JSON
        low_json = bp_dir / sav_json.name.replace("-savgol-", "-lowess-")
        # Smoothed CSVs
        sav_csv = _find_processed_csv(sav_dir, well_id, date)
        low_csv = _find_processed_csv(low_dir, well_id, date)
        # Raw
        raw_csv = _find_raw_csv(raw_dir, well_id, date)

        missing = []
        if not low_json.exists(): missing.append(f"lowess JSON ({low_json.name})")
        if sav_csv is None:       missing.append("savgol processed CSV")
        if low_csv is None:       missing.append("lowess processed CSV")
        if raw_csv is None:       missing.append("raw YSI CSV")

        if missing:
            skipped.append(SkippedRun(
                well_id=well_id, date=date,
                reason="missing companion files", missing=missing,
            ))
            continue

        bundles.append(WellRunBundle(
            well_id=well_id, date=date,
            raw_csv=raw_csv,
            savgol_csv=sav_csv,
            lowess_csv=low_csv,
            savgol_json=sav_json,
            lowess_json=low_json,
        ))

    return bundles, skipped


# ─────────────────────────────────────────────────────────────────────────
#  Rendering one bundle
# ─────────────────────────────────────────────────────────────────────────
def render_bundle(
    bundle: WellRunBundle,
    *,
    fig_root: Path,
    trial: str = "trial_1",
) -> tuple[int, int]:
    """Render all N figures for one (well_id, date).

    Returns (n_rendered, n_max) — number of figures actually written and
    the maximum N found in the JSON.
    """
    raw   = load_ysi_csv(bundle.raw_csv,   standardise=True)
    sav   = pd.read_csv(bundle.savgol_csv)
    low   = pd.read_csv(bundle.lowess_csv)
    bic_s = load_bic_json(bundle.savgol_json)
    bic_l = load_bic_json(bundle.lowess_json)

    # max_n_breakpoints from the JSON itself
    df_t = pd.DataFrame(bic_s[trial]["df"])
    n_max = int(df_t["n_breakpoints"].max())

    out_dir = fig_root / f"{bundle.well_id}_compare_N1to{n_max}"
    out_dir.mkdir(parents=True, exist_ok=True)

    z_raw = raw["depth_m"].to_numpy()
    EC_raw = raw["sec_uS_cm"].to_numpy()
    z_sav = sav["depth_m"].to_numpy()
    EC_sav = sav["sec_uS_cm"].to_numpy()
    z_low = low["depth_m"].to_numpy()
    EC_low = low["sec_uS_cm"].to_numpy()

    rendered = 0
    for n in range(1, n_max + 1):
        plot_breakpoints_compare_methods(
            z_raw=z_raw, EC_raw=EC_raw,
            z_left=z_sav, EC_left=EC_sav,
            bic_data_left=bic_s, label_left="savgol",
            z_right=z_low, EC_right=EC_low,
            bic_data_right=bic_l, label_right="lowess",
            n_breakpoints=n,
            trial=trial,
            output_path=out_dir / f"{bundle.well_id}_{bundle.date}_N{n:02d}.png",
            title=f"{bundle.well_id} {bundle.date}",
        )
        rendered += 1

    return rendered, n_max


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default="2022_02",
                   help="Campaign tag (default: 2022_02).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to specific well_id(s), e.g. --only LRS70D AW6D")
    p.add_argument("--well-type", default=None, choices=["D", "S", "O"],
                   help="Restrict to a single well type (D, S, or O).")
    p.add_argument("--trial", default="trial_1",
                   help="Which trial in the JSON to use (default: trial_1).")
    p.add_argument("--bp-dir", default=None,
                   help="Override: data/breakpoints/<campaign>")
    p.add_argument("--raw-dir", default=None,
                   help="Override: data/raw/sec/<campaign>")
    p.add_argument("--proc-root", default=None,
                   help="Override: data/processed/sec/<campaign>")
    p.add_argument("--fig-root", default=None,
                   help="Override: results/figures/breakpoints/<campaign>")
    args = p.parse_args()

    bp_dir   = Path(args.bp_dir)   if args.bp_dir   else \
               Path(f"data/breakpoints/{args.campaign}")
    raw_dir  = Path(args.raw_dir)  if args.raw_dir  else \
               Path(f"data/raw/sec/{args.campaign}")
    proc_root = Path(args.proc_root) if args.proc_root else \
                Path(f"data/processed/sec/{args.campaign}")
    if args.fig_root:
        fig_root = Path(args.fig_root)
    else:
        from karst_analysis.io import resolve_figure_dir
        fig_root = resolve_figure_dir("breakpoints", campaigns=[args.campaign])

    sav_dir = proc_root / "savgol"
    low_dir = proc_root / "lowess"

    if not bp_dir.is_dir():
        print(f"ERROR: breakpoints dir not found: {bp_dir}", file=sys.stderr)
        return 1

    print()
    print("=" * 72)
    print(" REGENERATE BREAKPOINT FIGURES")
    print("=" * 72)
    print(f"  campaign     : {args.campaign}")
    print(f"  bp-dir       : {bp_dir}")
    print(f"  raw-dir      : {raw_dir}")
    print(f"  proc-root    : {proc_root}")
    print(f"  fig-root     : {fig_root}")
    if args.only:        print(f"  only wells   : {args.only}")
    if args.well_type:   print(f"  only type    : {args.well_type}")
    print(f"  trial        : {args.trial}")
    print("=" * 72)

    bundles, skipped = _discover_bundles(
        bp_dir=bp_dir, raw_dir=raw_dir,
        sav_dir=sav_dir, low_dir=low_dir,
        only_wells=set(args.only) if args.only else None,
        only_well_type=args.well_type,
    )

    print(f"\nFound {len(bundles)} renderable run(s); {len(skipped)} skipped.")

    total_figs = 0
    for b in bundles:
        try:
            rendered, n_max = render_bundle(b, fig_root=fig_root, trial=args.trial)
            total_figs += rendered
            print(f"  ✓ {b.well_id} {b.date}: {rendered} figures (N=1..{n_max})")
        except Exception as e:
            print(f"  ✗ {b.well_id} {b.date}: {type(e).__name__}: {e}")
            skipped.append(SkippedRun(
                well_id=b.well_id, date=b.date,
                reason=f"render error: {type(e).__name__}: {e}",
            ))

    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"  Rendered : {len(bundles)} run(s) → {total_figs} figure(s)")
    print(f"  Skipped  : {len(skipped)}")

    if skipped:
        print("\nSkipped detail:")
        for s in skipped:
            line = f"  - {s.well_id} {s.date}: {s.reason}"
            if s.missing:
                line += "\n      missing: " + ", ".join(s.missing)
            print(line)

    return 0 if total_figs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
