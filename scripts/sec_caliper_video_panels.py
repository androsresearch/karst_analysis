"""Build SEC + caliper × video-log × Ardaman panels in batch.

Renders one PNG per (well, smoothing, n, trial) combination. The script
has two operating modes:

1. **Jobs mode** — preferred for the chapter. Pass a YAML jobs file
   (same format as ``scripts/slopes_batch.py``) and one panel is
   produced per job, honouring the per-well trial/method/N choices::

        uv run python scripts/sec_caliper_video_panels.py \
            --jobs config/slopes_jobs_2022_02.yml

   For the slopes-CSV–driven mixing-zone colouring (red diamond for
   TOP MZ, purple for BOTTOM MZ) to appear, the matching slopes CSVs
   must already be on disk under ``data/slopes/<campaign>/`` — run
   ``scripts/slopes_batch.py`` with the same YAML first.

2. **Legacy grid mode** — sweep ``wells × smoothings × [n_min..n_max]``
   for a single fixed trial (default ``trial_1``). Useful for
   sensitivity inspection::

        uv run python scripts/sec_caliper_video_panels.py \
            --wells LRS70D --smoothing savgol --n-min 1 --n-max 10 \
            --trial trial_1

Filename convention
-------------------
Every PNG carries an explicit ``__t{idx}`` suffix so trials at the same
N never collide::

    results/figures/convergence/sec_caliper_video/<campaign>/<well>/
        <well>_<date>__<smoothing>__N<nn>__t<idx>.png

Prerequisites
-------------
The SEC and caliper pipelines must have run first:

    uv run python scripts/preprocess_batch.py
    uv run python scripts/breakpoints_batch.py
    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_run_pipeline.py

Plus (for mixing-zone colouring in jobs mode):

    uv run python scripts/slopes_batch.py --jobs <same YAML>

This script reads:

    data/processed/sec/<campaign>/<smoothing>/{well}_*.csv
    data/breakpoints/<campaign>/{well}_*__bp-{smoothing}-*.json
    data/slopes/<campaign>/{well}_*__slopes-{method}-N{n}-t{idx}.csv
    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/raw/caliper/concatenate_caliper_all.csv
    data/raw/videolog/Priority_Ewan_video_logs_v2.xlsx
    data/raw/drilling/ardaman_lithology.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    WELLS, build_all_sec_caliper_video_panels,
)
from karst_analysis.io import resolve_figure_dir
from karst_analysis.sec.jobs_io import load_jobs_file


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Jobs mode ────────────────────────────────────────────────────
    p.add_argument(
        "--jobs", default=None,
        help=(
            "YAML jobs file (same schema as slopes_batch.py). When given, "
            "the grid-mode flags --wells/--smoothing/--n-min/--n-max/--trial "
            "are ignored and the campaign is read from the YAML."
        ),
    )

    # ── Legacy grid-mode flags ──────────────────────────────────────
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
    p.add_argument("--trial", default="trial_1",
                   help=(
                       "Trial to use for every (well, smoothing, n) in "
                       "grid mode. Default: trial_1. Use \"best_bic\" to "
                       "let the loader pick the lowest-BIC trial at each N."
                   ))
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

    # ── Resolve mode ────────────────────────────────────────────────
    if args.jobs is not None:
        jobs_path = Path(args.jobs)
        if not jobs_path.is_file():
            print(f"error: --jobs file not found: {jobs_path}", file=sys.stderr)
            return 2
        campaign, _default_threshold, job_objs = load_jobs_file(jobs_path)
        # The Job dataclass is converted to dicts to keep the
        # convergence-package API independent of karst_analysis.sec.
        jobs = [
            {"well": j.well, "method": j.method,
             "trial": j.trial, "n": j.n}
            for j in job_objs
        ]
        # Warn if grid-mode flags were also passed but ignored.
        legacy_explicit = (
            args.wells is not None
            or args.smoothing != ["savgol", "lowess"]
            or args.n_min != 1
            or args.n_max != 10
            or args.trial != "trial_1"
        )
        if legacy_explicit:
            print(
                "warning: --jobs given; --wells/--smoothing/--n-min/"
                "--n-max/--trial are ignored.",
                file=sys.stderr,
            )
    else:
        campaign = args.campaign
        jobs = None
        if args.n_min < 1:
            print("error: --n-min must be >= 1", file=sys.stderr)
            return 2
        if args.n_max < args.n_min:
            print("error: --n-max must be >= --n-min", file=sys.stderr)
            return 2

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else resolve_figure_dir("convergence/sec_caliper_video",
                                campaigns=[campaign])
    )

    print()
    print("=" * 72)
    print(" SEC + CALIPER × VIDEO-LOG × ARDAMAN PANELS")
    print("=" * 72)
    print(f"  output-dir : {output_dir}")
    print(f"  campaign   : {campaign}")
    if jobs is not None:
        print(f"  mode       : jobs ({len(jobs)} job(s) from {args.jobs})")
        n_panels_max = len(jobs)
    else:
        print(f"  mode       : grid")
        print(f"  wells      : {args.wells if args.wells else 'ALL'}")
        print(f"  smoothing  : {args.smoothing}")
        print(f"  N range    : [{args.n_min}, {args.n_max}]")
        print(f"  trial      : {args.trial}")
        n_wells = len(args.wells) if args.wells else len(WELLS)
        n_panels_max = (
            n_wells * len(args.smoothing) * (args.n_max - args.n_min + 1)
        )
    print(f"  expected   : up to {n_panels_max} panels")
    print("=" * 72)

    paths = build_all_sec_caliper_video_panels(
        wells=args.wells,
        smoothings=tuple(args.smoothing),
        n_min=args.n_min,
        n_max=args.n_max,
        trial=args.trial,
        jobs=jobs,
        campaign=campaign,
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
