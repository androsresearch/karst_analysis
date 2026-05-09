"""Run the full caliper pipeline: baseline + detection + per-sample export + panel.

Reads the master caliper CSV and the noise JSON, runs the cumulative-min
baseline + breakout detection on each priority well, and writes:

    data/processed/caliper/priority_wells_cumulative_min_v2_perpoint.csv
    data/processed/caliper/priority_wells_cumulative_min_v2_zones.csv
    results/figures/caliper/priority_wells_cumulative_min_v2_panel.png

Usage
-----
    uv run python scripts/caliper_run_pipeline.py

    # Override individual paths
    uv run python scripts/caliper_run_pipeline.py \\
        --master data/raw/caliper/concatenate_caliper_all.csv \\
        --noise  data/processed/caliper/noise_comparison.json \\
        --out-dir data/processed/caliper/ \\
        --fig-dir results/figures/caliper/

Prerequisites
-------------
``scripts/caliper_estimate_noise.py`` must have been run first to produce
the noise JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from karst_analysis.caliper.io import load_master_caliper, DEFAULT_MASTER_CSV
from karst_analysis.caliper.pipeline import (
    process_many_wells, perpoint_dataframe, zones_dataframe,
)
from karst_analysis.caliper.viz import plot_priority_wells_panel
from karst_analysis.caliper.config import PRIORITY_WELLS
from karst_analysis.io import resolve_figure_dir


DEFAULT_NOISE_JSON = Path("data/processed/caliper/noise_comparison.json")
DEFAULT_OUT_DIR    = Path("data/processed/caliper")
# Caliper is a pre-casing technique with no campaign concept — its
# figure goes directly under results/figures/caliper/ (v13 convention).
DEFAULT_FIG_DIR    = resolve_figure_dir("caliper")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=str(DEFAULT_MASTER_CSV))
    p.add_argument("--noise",  default=str(DEFAULT_NOISE_JSON))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="Where to put the per-sample and zones CSVs.")
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR),
                   help="Where to put the panel figure.")
    p.add_argument("--no-panel", action="store_true",
                   help="Skip generating the panel figure (CSVs only).")
    p.add_argument("--float-format", default="%.4f",
                   help="Float format for the per-sample CSV (default: '%%.4f' "
                        "to match the legacy pipeline).")
    args = p.parse_args()

    master_path = Path(args.master)
    noise_path = Path(args.noise)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    if not master_path.exists():
        print(f"ERROR: master CSV not found: {master_path}", file=sys.stderr)
        return 1
    if not noise_path.exists():
        print(f"ERROR: noise JSON not found: {noise_path}", file=sys.stderr)
        print("Run scripts/caliper_estimate_noise.py first.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" CALIPER PIPELINE")
    print("=" * 72)
    print(f"  master   : {master_path}")
    print(f"  noise    : {noise_path}")
    print(f"  out-dir  : {out_dir}")
    if not args.no_panel:
        print(f"  fig-dir  : {fig_dir}")
    print(f"  wells    : {PRIORITY_WELLS}")
    print("=" * 72)

    # 1) Load inputs
    df = load_master_caliper(master_path)
    with open(noise_path) as f:
        noise = json.load(f)
    sigma = float(noise["AW5O"]["sigma_MAD_cm"])
    print(f"\n  sigma_inst (AW5O sigma_MAD) = {sigma:.6f} cm")

    # 2) Run pipeline
    results = process_many_wells(df, sigma)

    # 3) Per-sample CSV
    perpoint = perpoint_dataframe(results)
    perpoint_path = out_dir / "priority_wells_cumulative_min_v2_perpoint.csv"
    perpoint.to_csv(perpoint_path, index=False, float_format=args.float_format)
    print(f"\n  ✓ {perpoint_path}  ({len(perpoint)} rows)")

    # 4) Zones CSV
    zones = zones_dataframe(results)
    zones_path = out_dir / "priority_wells_cumulative_min_v2_zones.csv"
    zones.to_csv(zones_path, index=False)
    print(f"  ✓ {zones_path}  ({len(zones)} zones)")

    # 5) Panel figure
    if not args.no_panel:
        fig_path = fig_dir / "priority_wells_cumulative_min_v2_panel.png"
        plot_priority_wells_panel(
            results, sigma, output_path=fig_path,
            well_order=PRIORITY_WELLS,
        )
        print(f"  ✓ {fig_path}")

    # 6) Summary
    print("\n  Per-well severity counts:")
    print("  " + "-" * 60)
    counts = perpoint.groupby(["well", "severity_per_sample"]).size().unstack(fill_value=0)
    for sev in ["none", "mild", "moderate", "severe"]:
        if sev not in counts.columns:
            counts[sev] = 0
    counts = counts[["none", "mild", "moderate", "severe"]]
    for w in PRIORITY_WELLS:
        if w in counts.index:
            row = counts.loc[w]
            print(f"  {w:<8} none={row['none']:>4} "
                  f"mild={row['mild']:>4} "
                  f"mod={row['moderate']:>3} "
                  f"sev={row['severe']:>3}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
