"""Build INTERACTIVE site panels  (v14).

Plotly counterpart to ``scripts/site_panels.py`` (v12). For each
priority site, produces a self-contained ``.html`` file with:

* shared depth axis between caliper and SEC (zooming one zooms both)
* WebGL-rendered SEC traces (responsive even with thousands of points)
* embedded Plotly.js (no internet required to open the file)
* legend toggling — click a campaign to hide/show all its wells

Defaults
--------
By default this script renders the SIX official campaigns of the
thesis (2011_05, 2022_02, 2022_08, 2023_08, 2025_02, 2025_11) for
all 5 priority sites (AW5, AW6, BW3, LRS69, LRS70).

Output
------
::

    results/figures/convergence/site_panel_interactive/<sub>/
        AW5_site_panel.html
        AW6_site_panel.html
        BW3_site_panel.html
        LRS69_site_panel.html
        LRS70_site_panel.html

where ``<sub>`` is the campaign name (single campaign) or
``multi_<N>c`` (multi-campaign overlay) following the v13 convention.

Usage
-----
    # Default (6 campaigns × 5 sites):
    uv run python scripts/site_panels_interactive.py

    # Subset of campaigns:
    uv run python scripts/site_panels_interactive.py --campaigns 2022_02 2022_08 2023_08

    # Subset of sites:
    uv run python scripts/site_panels_interactive.py --sites AW5 LRS69

    # Only D wells:
    uv run python scripts/site_panels_interactive.py --well-types D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from karst_analysis.convergence import (
    InteractiveSitePanelConfig,
    build_all_site_panels_interactive,
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
        help=f"Field campaigns to overlay. Default: {DEFAULT_CAMPAIGNS}.",
    )
    p.add_argument(
        "--sites", nargs="+", default=None,
        help=("Subset of sites to render (default: "
              + " ".join(_all_priority_sites()) + ")."),
    )
    p.add_argument(
        "--well-types", nargs="+", default=["D", "O", "S"],
        choices=["D", "O", "S"],
        help="Well types to overlay on the SEC axis (default: D O S).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help=("Where HTMLs are written. Default: "
              "results/figures/convergence/site_panel_interactive/<sub>/"),
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
        "--sec-linear", action="store_true",
        help="Use linear scale on the SEC axis (default: log).",
    )
    p.add_argument(
        "--sec-min", type=float, default=200.0,
        help=("Minimum SEC value plotted (default: 200 uS/cm). Filters "
              "out instrumental in-air readings. Set to 0 to disable."),
    )

    args = p.parse_args()

    cfg = InteractiveSitePanelConfig(
        sec_log_x=not args.sec_linear,
        sec_min_uS_cm=args.sec_min,
    )

    written = build_all_site_panels_interactive(
        campaigns=args.campaigns,
        sites=args.sites,
        well_types=args.well_types,
        perpoint_csv=args.perpoint,
        master_caliper_csv=args.master_caliper,
        project_root=args.project_root,
        output_dir=args.output_dir,
        config=cfg,
    )
    if args.output_dir is None:
        resolved = resolve_figure_dir(
            "convergence/site_panel_interactive",
            campaigns=args.campaigns,
        )
        print(f"\nDone. {len(written)} HTML(s) written to {resolved}/")
    else:
        print(f"\nDone. {len(written)} HTML(s) written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
