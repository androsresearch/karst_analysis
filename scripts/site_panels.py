"""Build the SEC × caliper panels grouped by SITE  (v12).

Runs ``karst_analysis.convergence.build_all_site_panels`` and saves
one PNG per priority site plus a master 1×N figure where every site
is a column-pair (caliper + SEC).

Layout per site:
    column 1 (caliper) : per-sample severity bands + caliper signal
                         for the site's D well (the only one with a
                         caliper run).
    column 2 (SEC)     : raw YSI traces for ALL wells of the site
                         (D, O, S where they exist) across ALL the
                         requested campaigns. Each trace is identified
                         by:
                            colour     = campaign (fixed thesis palette)
                            line-style = well type (D solid, O dotted,
                                         S dashed)

Defaults
--------
By default this script renders the SIX official campaigns of the
thesis: 2011_05, 2022_02, 2022_08, 2023_08, 2025_02, 2025_11. Pass
``--campaigns`` to override.

Differences w.r.t. ``sec_caliper_panels.py`` (v11)
--------------------------------------------------
v11 produces one panel per priority *well* (one column-pair per pozo
D, with a single SEC column showing only that pozo's casts). v12
produces one panel per *site* (one column-pair per sitio, with the
SEC column overlaying every well type of that site across every
campaign). Both scripts coexist; see ``sec_caliper_panels.py`` for
the per-well view.

Prerequisites
-------------
Per-sample severity CSV (produced by the caliper pipeline):

    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_run_pipeline.py

This script then reads:

    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/raw/caliper/concatenate_caliper_all.csv
    data/raw/sec/<campaign>/[<well_type>/]*.csv      (any layout)
    data/metadata/wells.csv

Usage
-----
    # Default: all 6 official campaigns, all 5 sites:
    uv run python scripts/site_panels.py

    # Subset of campaigns:
    uv run python scripts/site_panels.py --campaigns 2022_02 2022_08 2023_08

    # A subset of sites:
    uv run python scripts/site_panels.py --sites AW5 LRS69

    # Only D wells (no O / S overlay):
    uv run python scripts/site_panels.py --well-types D

    # Skip the master figure:
    uv run python scripts/site_panels.py --no-master

    # Custom output dir:
    uv run python scripts/site_panels.py --output-dir my_panels/

    # Linear SEC (instead of the default log scale):
    uv run python scripts/site_panels.py --sec-linear

    # Disable the in-air noise filter (default 200 µS/cm):
    uv run python scripts/site_panels.py --sec-min 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    SitePanelConfig,
    build_all_site_panels,
)
from karst_analysis.convergence.site_panel import _all_priority_sites
from karst_analysis.io import resolve_figure_dir


DEFAULT_CAMPAIGNS = [
    "2011_05", "2022_02", "2022_08", "2023_08", "2025_02", "2025_11",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--campaigns", nargs="+", default=DEFAULT_CAMPAIGNS,
        help=("Field campaigns to overlay on the SEC axis. "
              f"Default: {DEFAULT_CAMPAIGNS}."),
    )
    p.add_argument(
        "--sites", nargs="+", default=None,
        help=("Subset of sites to render (default: all priority sites: "
              + " ".join(_all_priority_sites()) + ")."),
    )
    p.add_argument(
        "--well-types", nargs="+", default=["D", "O", "S"],
        choices=["D", "O", "S"],
        help="Well types to overlay on the SEC axis (default: D O S).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help=("Where PNGs are written. Default: "
              "results/figures/convergence/site_panel/<campaign-or-multi_Nc>/"),
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
        "--sec-linear", action="store_true",
        help="Use linear scale on the SEC axis (default: log).",
    )
    p.add_argument(
        "--sec-min", type=float, default=200.0,
        help=("Minimum SEC value plotted (default: 200 µS/cm). Filters "
              "out instrumental in-air readings. Set to 0 to disable."),
    )

    args = p.parse_args()

    cfg = SitePanelConfig(
        sec_log_x=not args.sec_linear,
        sec_min_uS_cm=args.sec_min,
    )

    written = build_all_site_panels(
        campaigns=args.campaigns,
        sites=args.sites,
        well_types=args.well_types,
        perpoint_csv=args.perpoint,
        master_caliper_csv=args.master_caliper,
        project_root=args.project_root,
        output_dir=args.output_dir,
        config=cfg,
        build_master=not args.no_master,
    )
    if args.output_dir is None:
        resolved = resolve_figure_dir("convergence/site_panel",
                                      campaigns=args.campaigns)
        print(f"\nDone. {len(written)} figure(s) written to {resolved}/")
    else:
        print(f"\nDone. {len(written)} figure(s) written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
