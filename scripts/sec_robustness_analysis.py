"""Run the SEC breakpoint robustness analysis for one campaign.

Pools breakpoints across (smoothing, N) combinations, clusters them by
depth (single-linkage with threshold δ), and reports per-cluster
persistence / agreement scores. Also runs the same analysis at three δ
values for sensitivity, and emits BIC curves to help decide N.

Outputs (v13)
-------------
CSVs stay under ``results/sec_robustness/<campaign>/``::

    robustness_clusters.csv      — one row per cluster, all wells
    robustness_bp_assignments.csv — one row per BP, with cluster_id
    robustness_summary.csv       — BIC-optimal N per (well, smoothing)
    sensitivity_clusters.csv     — clusters at δ ∈ {0.3, 0.5, 1.0} m

Figures move to ``results/figures/sec_robustness/<campaign>/``::

    robustness_<well>.png    — per-well diagnostic panel (5 files)
    sensitivity_<well>.png   — per-well δ-sensitivity (5 files)
    bic_curves.png           — BIC(N) for all wells, both smoothings

Prerequisites
-------------
The SEC preprocessing and breakpoint detection batches must have run:

    uv run python scripts/preprocess_batch.py
    uv run python scripts/breakpoints_batch.py

Usage
-----
    # All priority wells, default δ=0.5, sensitivity at {0.3, 0.5, 1.0}
    uv run python scripts/sec_robustness_analysis.py

    # One well only
    uv run python scripts/sec_robustness_analysis.py --wells LRS70D

    # Custom δ values and N range
    uv run python scripts/sec_robustness_analysis.py \\
        --delta 0.4 --sensitivity-deltas 0.2 0.4 0.8 --n-min 2 --n-max 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from karst_analysis.convergence import WELLS
from karst_analysis.sec.robustness import (
    DEFAULT_DELTA_M,
    SENSITIVITY_DELTAS_M,
    compute_robustness,
    compute_robustness_sensitivity,
    plot_robustness_panel,
    plot_delta_sensitivity,
    plot_bic_curves,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config (default: config/pipeline.yml). "
                        "CLI args below, when given, override the config.")
    p.add_argument("--campaign", default=None,
                   help="Field campaign. If omitted, read from config.")
    p.add_argument("--wells", nargs="+", default=None,
                   choices=list(WELLS.keys()),
                   help="Subset of wells (default: all).")
    p.add_argument("--smoothing", nargs="+", default=None,
                   choices=["savgol", "lowess"],
                   help="Smoothings to pool. If omitted, read from config.")
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--delta", type=float, default=None,
                   help="Linkage threshold for the main analysis (m).")
    p.add_argument("--sensitivity-deltas", nargs="+", type=float,
                   default=None,
                   help="δ values for the sensitivity scan.")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: results/sec_robustness/<campaign>/).")
    p.add_argument("--skip-figures", action="store_true",
                   help="Skip the figure generation (CSVs only).")
    args = p.parse_args()

    # ── Load config and apply CLI overrides ──
    from karst_analysis.config import (
        ConfigError, default_config_path, load_config,
    )
    if args.config is not None:
        cfg_path = Path(args.config)
    else:
        candidate = default_config_path().parent / "pipeline.yml"
        cfg_path = candidate if candidate.exists() else None

    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        print(f"ERROR loading config: {exc}", file=sys.stderr)
        return 2

    rb = cfg["robustness"]
    campaign = args.campaign if args.campaign is not None else cfg["campaign"]
    smoothings = tuple(args.smoothing) if args.smoothing else tuple(rb["smoothings"])
    n_min = args.n_min if args.n_min is not None else rb["n_min"]
    n_max = args.n_max if args.n_max is not None else rb["n_max"]
    delta = args.delta if args.delta is not None else rb["delta_m"]
    sens_deltas = (list(args.sensitivity_deltas)
                   if args.sensitivity_deltas
                   else list(rb["sensitivity_deltas_m"]))

    out_dir = (Path(args.output_dir)
               if args.output_dir is not None
               else Path("results") / "sec_robustness" / campaign)
    if not args.skip_figures:
        from karst_analysis.io import resolve_figure_dir
        fig_dir = resolve_figure_dir("sec_robustness", campaigns=[campaign])
    else:
        fig_dir = out_dir / "figures"  # never created, kept for shape compat
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    target_wells = args.wells if args.wells else list(WELLS.keys())
    print()
    print("=" * 72)
    print(" SEC BREAKPOINT ROBUSTNESS ANALYSIS")
    print("=" * 72)
    print(f"  config              : {cfg_path if cfg_path else 'defaults only'}")
    print(f"  campaign            : {campaign}")
    print(f"  wells               : {target_wells}")
    print(f"  smoothings          : {list(smoothings)}")
    print(f"  N range             : [{n_min}, {n_max}]")
    print(f"  delta (main)        : {delta} m")
    print(f"  delta (sensitivity) : {sens_deltas}")
    print(f"  output_dir          : {out_dir}")
    print("=" * 72)
    print()

    all_clusters: list[pd.DataFrame] = []
    all_bp_assignments: list[pd.DataFrame] = []
    all_bic_summary: list[pd.DataFrame] = []
    all_sensitivity: list[pd.DataFrame] = []

    for well in target_wells:
        print(f"[{well}]")
        try:
            res = compute_robustness(
                well, campaign=campaign,
                smoothings=smoothings,
                n_range=(n_min, n_max),
                delta_m=delta,
            )
        except ValueError as exc:
            print(f"  ✗ {exc}")
            continue
        n_clusters = len(res.clusters)
        n_bps = len(res.bp_records)
        print(f"  ✓ {n_bps} BPs pooled, {n_clusters} clusters at δ={delta}m")
        clusters = res.clusters.copy()
        clusters.insert(0, "well_id", well)
        clusters.insert(1, "delta_m", delta)
        all_clusters.append(clusters)
        all_bp_assignments.append(res.bp_records)
        all_bic_summary.append(res.bic_summary)

        sens_df = compute_robustness_sensitivity(
            well, campaign=campaign,
            smoothings=smoothings,
            n_range=(n_min, n_max),
            deltas_m=tuple(sens_deltas),
        )
        if not sens_df.empty:
            all_sensitivity.append(sens_df)

        if not args.skip_figures:
            try:
                fig = plot_robustness_panel(
                    res, well_id=well,
                    output_path=fig_dir / f"robustness_{well}.png",
                )
                plt.close(fig)
                fig = plot_delta_sensitivity(
                    sens_df, well_id=well,
                    output_path=fig_dir / f"sensitivity_{well}.png",
                )
                plt.close(fig)
                print(f"  ✓ figures saved")
            except Exception as exc:
                print(f"  ✗ figure error: {exc}")

    if not args.skip_figures and target_wells:
        try:
            fig = plot_bic_curves(
                target_wells, campaign=campaign,
                output_path=fig_dir / "bic_curves.png",
            )
            plt.close(fig)
            print(f"\n✓ BIC curves saved")
        except Exception as exc:
            print(f"\n✗ BIC curves error: {exc}")

    if all_clusters:
        df = pd.concat(all_clusters, ignore_index=True)
        df.to_csv(out_dir / "robustness_clusters.csv", index=False)
        print(f"\n✓ {out_dir / 'robustness_clusters.csv'} ({len(df)} rows)")
    if all_bp_assignments:
        df = pd.concat(all_bp_assignments, ignore_index=True)
        df.to_csv(out_dir / "robustness_bp_assignments.csv", index=False)
        print(f"✓ {out_dir / 'robustness_bp_assignments.csv'} ({len(df)} rows)")
    if all_bic_summary:
        df = pd.concat(all_bic_summary, ignore_index=True)
        df.to_csv(out_dir / "robustness_summary.csv", index=False)
        print(f"✓ {out_dir / 'robustness_summary.csv'} ({len(df)} rows)")
    if all_sensitivity:
        df = pd.concat(all_sensitivity, ignore_index=True)
        df.to_csv(out_dir / "sensitivity_clusters.csv", index=False)
        print(f"✓ {out_dir / 'sensitivity_clusters.csv'} ({len(df)} rows)")

    print()
    print("=" * 72)
    print(" DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
