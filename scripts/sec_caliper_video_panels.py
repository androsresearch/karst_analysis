"""Build SEC + caliper × video-log × Ardaman panels in batch.

Renders one PNG per (well, smoothing, n) combination. With 5 priority
wells × 2 smoothings × 10 N values, a full run produces 100 PNGs in
``results/figures/convergence/sec_caliper_video/<well>/``.

Prerequisites
-------------
The SEC and caliper pipelines must have run first:

    uv run python scripts/preprocess_batch.py
    uv run python scripts/breakpoints_batch.py
    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_run_pipeline.py

Then this script reads:

    data/processed/sec/<campaign>/<smoothing>/{well}_*.csv
    data/breakpoints/<campaign>/{well}_*__bp-{smoothing}-*.json
    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/raw/caliper/concatenate_caliper_all.csv
    data/raw/videolog/Priority_Ewan_video_logs_v2.xlsx
    data/raw/drilling/ardaman_lithology.csv

Usage
-----
    # Full batch (5 wells × 2 smoothings × 10 N values = 100 PNGs)
    uv run python scripts/sec_caliper_video_panels.py

    # Just one well, both smoothings, all N
    uv run python scripts/sec_caliper_video_panels.py --wells LRS70D

    # All wells, savgol only, N from 1 to 5
    uv run python scripts/sec_caliper_video_panels.py \
        --smoothing savgol --n-min 1 --n-max 5

    # Combinations
    uv run python scripts/sec_caliper_video_panels.py \
        --wells LRS70D AW5D --smoothing savgol --n-min 3 --n-max 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    WELLS, build_all_sec_caliper_video_panels,
)
from karst_analysis.io import resolve_figure_dir


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wells", nargs="+", default=None,
                   choices=list(WELLS.keys()),
                   help="Subset of wells (default: all).")
    p.add_argument("--smoothing", nargs="+", default=["savgol", "lowess"],
                   choices=["savgol", "lowess"],
                   help="Subset of smoothings (default: both).")
    p.add_argument("--n-min", type=int, default=1,
                   help="Minimum N (default: 1).")
    p.add_argument("--n-max", type=int, default=10,
                   help="Maximum N inclusive (default: 10).")
    p.add_argument("--campaign", default="2022_02",
                   help="Field campaign (default: 2022_02).")
    p.add_argument("--output-dir", default=None,
                   help=("Output directory. Default: "
                         "results/figures/convergence/sec_caliper_video/<campaign>/"))
    p.add_argument("--perpoint",
                   help="Override per-sample caliper CSV path.")
    p.add_argument("--master-caliper",
                   help="Override master caliper CSV path.")
    p.add_argument("--video-xlsx",
                   help="Override video-log xlsx path.")
    p.add_argument("--ardaman-csv",
                   help="Override Ardaman lithology csv path.")
    args = p.parse_args()

    if args.n_min < 1:
        print("error: --n-min must be >= 1", file=sys.stderr)
        return 2
    if args.n_max < args.n_min:
        print("error: --n-max must be >= --n-min", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else resolve_figure_dir("convergence/sec_caliper_video",
                                campaigns=[args.campaign])
    )

    print()
    print("=" * 72)
    print(" SEC + CALIPER × VIDEO-LOG × ARDAMAN PANELS")
    print("=" * 72)
    print(f"  output-dir : {output_dir}")
    print(f"  wells      : {args.wells if args.wells else 'ALL'}")
    print(f"  smoothing  : {args.smoothing}")
    print(f"  N range    : [{args.n_min}, {args.n_max}]")
    print(f"  campaign   : {args.campaign}")
    n_wells = len(args.wells) if args.wells else len(WELLS)
    n_panels_max = n_wells * len(args.smoothing) * (args.n_max - args.n_min + 1)
    print(f"  expected   : up to {n_panels_max} panels")
    print("=" * 72)

    paths = build_all_sec_caliper_video_panels(
        wells=args.wells,
        smoothings=tuple(args.smoothing),
        n_min=args.n_min,
        n_max=args.n_max,
        campaign=args.campaign,
        output_dir=output_dir,
        perpoint_csv=args.perpoint,
        master_caliper_csv=args.master_caliper,
        video_xlsx=args.video_xlsx,
        ardaman_csv=args.ardaman_csv,
    )

    print()
    print("=" * 72)
    print(f" SUMMARY: {len(paths)} panel(s) written")
    print("=" * 72)
    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
