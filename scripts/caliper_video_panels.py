"""Build the caliper × video-log × Ardaman panels for the priority wells.

Runs ``karst_analysis.convergence.build_all_caliper_video_panels`` and
saves one PNG per well in ``results/figures/convergence/caliper_video/``.

Prerequisites
-------------
The per-sample CSV must have been generated already:

    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_run_pipeline.py

Then this script reads:

    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/raw/caliper/concatenate_caliper_all.csv
    data/raw/videolog/Priority_Ewan_video_logs_v2.xlsx
    data/raw/drilling/ardaman_lithology.csv

Usage
-----
    uv run python scripts/caliper_video_panels.py
    uv run python scripts/caliper_video_panels.py --wells LRS70D AW6D
    uv run python scripts/caliper_video_panels.py --output-dir my_panels/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    WELLS, build_all_caliper_video_panels,
)
from karst_analysis.io import resolve_figure_dir


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--perpoint",
                   help="Override per-sample caliper CSV path.")
    p.add_argument("--master-caliper",
                   help="Override master caliper CSV path.")
    p.add_argument("--video-xlsx",
                   help="Override video-log xlsx path.")
    p.add_argument("--ardaman-csv",
                   help="Override Ardaman lithology csv path.")
    p.add_argument("--output-dir", default=None,
                   help=("Output directory. Default: "
                         "results/figures/convergence/caliper_video/ "
                         "(no campaign subfolder — caliper and video are "
                         "pre-casing techniques)."))
    p.add_argument("--wells", nargs="+", default=None,
                   choices=list(WELLS.keys()),
                   help="Subset of wells to render (default: all).")
    args = p.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else resolve_figure_dir("convergence/caliper_video")
    )

    print()
    print("=" * 72)
    print(" CALIPER × VIDEO-LOG × ARDAMAN PANELS")
    print("=" * 72)
    print(f"  output-dir : {output_dir}")
    print(f"  wells      : {args.wells if args.wells else 'ALL'}")
    print("=" * 72)
    print()

    paths = build_all_caliper_video_panels(
        perpoint_csv=args.perpoint,
        video_xlsx=args.video_xlsx,
        ardaman_csv=args.ardaman_csv,
        master_caliper_csv=args.master_caliper,
        output_dir=output_dir,
        wells=args.wells,
    )

    print()
    print("=" * 72)
    print(f" SUMMARY: {len(paths)} panel(s) written")
    print("=" * 72)

    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
