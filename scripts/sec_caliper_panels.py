"""Build the SEC raw × caliper panels for the priority wells  (v11).

Runs ``karst_analysis.convergence.build_all_sec_caliper_panels`` and
saves one PNG per well plus a master 1×N figure in the chosen output
directory.

The SEC axis can overlay several campaigns:
    * Each campaign gets a distinguishable colour from the Plotly Dark24
      palette.
    * If a (well, campaign) pair has no explicit vadose row in
      ``data/metadata/wells.csv`` and the raw CSV doesn't carry the
      ``Depth from GL (m)`` column, the loader falls back to the
      well's value in the reference campaign (``2022_02`` by default)
      and the legend marks that campaign with ``*``.

Prerequisites
-------------
Per-sample severity CSV (produced by the caliper pipeline):

    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_run_pipeline.py

This script then reads:

    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/raw/caliper/concatenate_caliper_all.csv
    data/raw/sec/<campaign>/*.csv             (one folder per campaign)
    data/metadata/wells.csv                     (vadose lookup)

Usage
-----
    # Default: just feb-2022 (mimics v10 behaviour):
    uv run python scripts/sec_caliper_panels.py

    # Multiple campaigns overlaid on the SEC axis:
    uv run python scripts/sec_caliper_panels.py --campaigns 2022_02 2022_06 2023_02

    # A subset of wells:
    uv run python scripts/sec_caliper_panels.py --wells AW6D LRS70D

    # Skip the master figure:
    uv run python scripts/sec_caliper_panels.py --no-master

    # Custom output dir:
    uv run python scripts/sec_caliper_panels.py --output-dir my_panels/

    # Log scale on SEC:
    uv run python scripts/sec_caliper_panels.py --sec-log-x
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    WELLS,
    SecCaliperPanelConfig,
    build_all_sec_caliper_panels,
)
from karst_analysis.io import resolve_figure_dir


DEFAULT_CAMPAIGNS = ["2022_02"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--campaigns", nargs="+", default=DEFAULT_CAMPAIGNS,
        help=("One or more field campaigns to overlay on the SEC axis. "
              f"Default: {DEFAULT_CAMPAIGNS}. Example: --campaigns 2022_02 "
              "2022_06 2023_02"),
    )
    p.add_argument(
        "--wells", nargs="+", default=None,
        help=("Subset of wells to render (default: all priority wells: "
              + " ".join(WELLS) + ")."),
    )
    p.add_argument(
        "--output-dir", default=None,
        help=("Where PNGs are written. Default: "
              "results/figures/convergence/sec_caliper_panel/<campaign-or-multi_Nc>/"),
    )
    p.add_argument(
        "--perpoint", default=None,
        help="Override per-sample caliper CSV path.",
    )
    p.add_argument(
        "--master-caliper", default=None,
        help="Override master caliper CSV path.",
    )
    p.add_argument(
        "--project-root", default=None,
        help="Override project root (defaults to current working directory).",
    )
    p.add_argument(
        "--no-master", action="store_true",
        help="Skip the master 1xN figure.",
    )
    p.add_argument(
        "--sec-log-x", action="store_true",
        help="Use log scale on the SEC axis (default: linear).",
    )

    args = p.parse_args()

    cfg = SecCaliperPanelConfig(sec_log_x=args.sec_log_x)

    if args.wells is not None:
        unknown = [w for w in args.wells if w not in WELLS]
        if unknown:
            print(f"Unknown wells: {unknown}. Known: {list(WELLS)}",
                  file=sys.stderr)
            return 2

    written = build_all_sec_caliper_panels(
        campaigns=args.campaigns,
        well_ids=args.wells,
        perpoint_csv=args.perpoint,
        master_caliper_csv=args.master_caliper,
        project_root=args.project_root,
        output_dir=args.output_dir,
        config=cfg,
        build_master=not args.no_master,
    )
    if args.output_dir is None:
        resolved = resolve_figure_dir("convergence/sec_caliper_panel",
                                      campaigns=args.campaigns)
        print(f"\nDone. {len(written)} figure(s) written to {resolved}/")
    else:
        print(f"\nDone. {len(written)} figure(s) written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
