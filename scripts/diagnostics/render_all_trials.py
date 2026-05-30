"""Throwaway: render breakpoint comparison figures for ALL trials.

For one campaign and every (well, N) pair, this script generates 3
side-by-side comparison figures (one per trial: trial_1, trial_2,
trial_3). Each figure shows savgol vs lowess at that N for that trial.

Total figures = wells × N × trials.

For 5 wells × 15 N × 3 trials = 225 figures.

This is intended for one-off visual inspection while the user decides
which (well, method, trial, N) combinations to feed to slopes_batch.py.

DO NOT KEEP THIS SCRIPT IN THE REPO LONG-TERM.
Once balance-spatial issues are resolved upstream, the trial-to-trial
spread should drop and this exploration becomes unnecessary.

Output structure
----------------
    results/figures/breakpoints_trial_inspection/<campaign>/<well>/
        N{nn}_trial{t}.png      e.g. N08_trial2.png

Usage
-----
    uv run python scripts/diagnostics/render_all_trials.py --campaign 2022_02
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from karst_analysis.corrections import get_vadose_thickness
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.viz import (
    load_bic_json,
    plot_breakpoints_compare_methods,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_all_trials")


# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Bundle:
    well_id: str
    date: str
    raw_csv: Path
    sav_csv: Path
    low_csv: Path
    sav_json: Path
    low_json: Path


def _parse_well_id(well_id: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)([DSO])$", well_id)
    if m is None:
        raise ValueError(f"Cannot split well_id '{well_id}'")
    return m.group(1), m.group(2)


def _find_raw_csv(raw_dir: Path, well_id: str, date: str) -> Optional[Path]:
    site, well_type = _parse_well_id(well_id)
    cands = list(raw_dir.glob(f"{site}_{well_type}_*_{date}.csv"))
    if not cands:
        cands = list(raw_dir.glob(f"{site}{well_type}_*_{date}.csv"))
    return cands[0] if cands else None


def _find_processed_csv(proc_dir: Path, well_id: str, date: str) -> Optional[Path]:
    if not proc_dir.exists():
        return None
    cands = sorted(proc_dir.glob(f"{well_id}_{date}__*.csv"))
    return cands[-1] if cands else None


def _discover_bundles(
    *, bp_dir: Path, raw_dir: Path, sav_dir: Path, low_dir: Path,
    only_wells: Optional[set[str]] = None,
) -> list[Bundle]:
    bundles: list[Bundle] = []
    sav_jsons = sorted(bp_dir.glob("*__bp-savgol-*.json"))
    for sj in sav_jsons:
        stem = sj.stem
        try:
            wd, _ = stem.split("__", 1)
            well_id, date = wd.rsplit("_", 1)
        except ValueError:
            logger.warning(f"unparseable: {sj.name}")
            continue
        if only_wells and well_id not in only_wells:
            continue

        lj = bp_dir / sj.name.replace("-savgol-", "-lowess-")
        sc = _find_processed_csv(sav_dir, well_id, date)
        lc = _find_processed_csv(low_dir, well_id, date)
        rc = _find_raw_csv(raw_dir, well_id, date)
        missing = [n for n, p in
                   [("lowess JSON", lj if lj.exists() else None),
                    ("savgol CSV", sc), ("lowess CSV", lc), ("raw CSV", rc)]
                   if p is None]
        if missing:
            logger.warning(f"  {well_id}_{date} missing: {missing}; skipping")
            continue
        bundles.append(Bundle(well_id, date, rc, sc, lc, sj, lj))
    return bundles


# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--bp-dir", type=Path, default=None)
    p.add_argument("--proc-dir", type=Path, default=None)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=15)
    p.add_argument("--trials", nargs="+", default=["trial_1", "trial_2", "trial_3"])
    p.add_argument("--only", nargs="+", default=None,
                   help="Filter by well_id, e.g. --only LRS70D AW6D")
    args = p.parse_args()

    bp_dir   = args.bp_dir   or Path(f"data/breakpoints/{args.campaign}")
    proc_dir = args.proc_dir or Path(f"data/processed/sec/{args.campaign}")
    raw_dir  = args.raw_dir  or Path(f"data/raw/sec/{args.campaign}")
    out_dir  = args.out_dir  or Path(
        f"results/figures/breakpoints_trial_inspection/{args.campaign}"
    )
    sav_dir = proc_dir / "savgol"
    low_dir = proc_dir / "lowess"

    print("=" * 72)
    print(" RENDER ALL TRIALS — breakpoint comparison figures (THROWAWAY)")
    print("=" * 72)
    print(f"  campaign  : {args.campaign}")
    print(f"  bp-dir    : {bp_dir}")
    print(f"  out-dir   : {out_dir}")
    print(f"  N range   : {args.n_min}..{args.n_max}")
    print(f"  trials    : {args.trials}")
    print(f"  only      : {args.only if args.only else 'all wells'}")
    print("=" * 72)

    only_wells = set(args.only) if args.only else None
    bundles = _discover_bundles(
        bp_dir=bp_dir, raw_dir=raw_dir, sav_dir=sav_dir, low_dir=low_dir,
        only_wells=only_wells,
    )
    if not bundles:
        logger.error("No bundles found.")
        return 1
    logger.info(f"Found {len(bundles)} well(s).")

    n_total = 0
    n_failed = 0
    for b in bundles:
        logger.info(f"[{b.well_id}_{b.date}] loading raw + smoothed + JSONs")
        df_raw = load_ysi_csv(b.raw_csv, standardise=True)
        z_raw  = df_raw["depth_m"].to_numpy()
        ec_raw = df_raw["sec_uS_cm"].to_numpy()

        df_sav = pd.read_csv(b.sav_csv)
        df_low = pd.read_csv(b.low_csv)
        z_sav, ec_sav = df_sav["depth_m"].to_numpy(), df_sav["sec_uS_cm"].to_numpy()
        z_low, ec_low = df_low["depth_m"].to_numpy(), df_low["sec_uS_cm"].to_numpy()

        bic_sav = load_bic_json(b.sav_json)
        bic_low = load_bic_json(b.low_json)

        # Look up vadose-zone thickness so figures render in BGL
        # (the canonical datum). If the well is missing from
        # wells.csv, fall back to 0.0 (water-table datum, honest label).
        try:
            vadose_offset_m = get_vadose_thickness(b.well_id)
        except (KeyError, FileNotFoundError) as exc:
            logger.warning(
                f"  no vadose value for {b.well_id} "
                f"({exc}); plotting in water-table datum"
            )
            vadose_offset_m = 0.0

        well_dir = out_dir / b.well_id
        well_dir.mkdir(parents=True, exist_ok=True)

        for n in range(args.n_min, args.n_max + 1):
            for trial in args.trials:
                fp = well_dir / f"N{n:02d}_{trial}.png"
                try:
                    plot_breakpoints_compare_methods(
                        z_raw=z_raw, EC_raw=ec_raw,
                        z_left=z_sav, EC_left=ec_sav,
                        bic_data_left=bic_sav, label_left="savgol",
                        z_right=z_low, EC_right=ec_low,
                        bic_data_right=bic_low, label_right="lowess",
                        n_breakpoints=n,
                        trial=trial,
                        output_path=fp,
                        title=f"{b.well_id} {b.date}  N={n}  {trial}",
                        vadose_offset_m=vadose_offset_m,
                    )
                    n_total += 1
                except Exception as e:
                    logger.warning(f"  N={n} {trial}: {e}")
                    n_failed += 1

        logger.info(f"  ✓ {b.well_id}: figures so far {n_total}, failed {n_failed}")

    print()
    print("=" * 72)
    print(f" SUMMARY: rendered {n_total} figures, failed {n_failed}")
    print(f" Output: {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
