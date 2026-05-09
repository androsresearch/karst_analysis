"""SEC ↔ caliper quantitative convergence (Idea 3).

For each robust SEC cluster, find the caliper anomalous zones it matches
and produce a convergence score per cluster. Aggregates per well.

Inputs (paths configurable via CLI; sensible defaults follow the repo
convention)
-----------------------------------------------------------------------
* SEC robust clusters (from sec_robustness_analysis.py):
    results/sec_robustness/<campaign>/robustness_clusters.csv
* Caliper zones (from the caliper pipeline):
    data/processed/caliper/priority_wells_cumulative_min_v2_zones.csv

Outputs (CSV only, under ``results/convergence/sec_caliper/<campaign>/``)
-----------------------------------------------------------
    cluster_matches.csv      — one row per analysed SEC cluster
    well_summary.csv         — one row per well
    unmatched_caliper_zones.csv — caliper zones with no SEC match

All matching parameters live in the YAML config under
``convergence.sec_caliper`` and are loaded via ``karst_analysis.config``.
The default config (config/pipeline_default.yml) reflects the choices
agreed for the thesis baseline. To experiment, copy to config/pipeline.yml
and override only the keys you want to change.

Usage
-----
    # Defaults
    uv run python scripts/sec_caliper_convergence.py

    # Custom config (e.g. tolerance sensitivity)
    uv run python scripts/sec_caliper_convergence.py --config config/sensitivity_tol1m.yml

    # Different campaign or input paths
    uv run python scripts/sec_caliper_convergence.py \\
        --campaign 2022_02 \\
        --sec-clusters results/sec_robustness/2022_02/robustness_clusters.csv \\
        --caliper-zones data/processed/caliper/priority_wells_cumulative_min_v2_zones.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from karst_analysis.config import (
    ConfigError, default_config_path, load_config, params_for_run_ledger,
)
from karst_analysis.convergence.sec_caliper_match import (
    compute_convergence,
    load_caliper_zones,
    load_sec_clusters,
)
from karst_analysis.runs import register_run


def _resolve_config_path(arg_value: str | None) -> Path | None:
    """Mimic the resolution used by sibling scripts."""
    if arg_value is not None:
        return Path(arg_value)
    candidate = default_config_path().parent / "pipeline.yml"
    return candidate if candidate.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config (default: config/pipeline.yml or "
                        "the frozen default if missing).")
    p.add_argument("--campaign", default=None,
                   help="Field campaign id, e.g. 2022_02. Defaults to "
                        "config['campaign'].")
    p.add_argument("--sec-clusters", default=None,
                   help="Path to robustness_clusters.csv. Default: "
                        "results/sec_robustness/<campaign>/robustness_clusters.csv")
    p.add_argument("--caliper-zones", default=None,
                   help="Path to caliper anomalous-zones CSV. Default: "
                        "data/processed/caliper/"
                        "priority_wells_cumulative_min_v2_zones.csv")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory.")
    p.add_argument("--no-ledger", action="store_true",
                   help="Skip writing an entry to results/runs.csv.")
    args = p.parse_args()

    # ── Load config ──
    cfg_path = _resolve_config_path(args.config)
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        print(f"ERROR loading config: {exc}", file=sys.stderr)
        return 2

    sub = cfg["convergence"]["sec_caliper"]
    campaign = args.campaign if args.campaign is not None else cfg["campaign"]

    # ── Resolve I/O paths ──
    sec_path = Path(args.sec_clusters) if args.sec_clusters else (
        Path("results") / "sec_robustness" / campaign / "robustness_clusters.csv"
    )
    cal_path = Path(args.caliper_zones) if args.caliper_zones else (
        Path("data") / "processed" / "caliper"
        / "priority_wells_cumulative_min_v2_zones.csv"
    )
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("results") / "convergence" / "sec_caliper" / campaign
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ──
    print()
    print("=" * 72)
    print(" SEC ↔ CALIPER QUANTITATIVE CONVERGENCE  (Idea 3)")
    print("=" * 72)
    print(f"  config                  : {cfg_path if cfg_path else 'defaults only'}")
    print(f"  campaign                : {campaign}")
    print(f"  sec_clusters            : {sec_path}")
    print(f"  caliper_zones           : {cal_path}")
    print(f"  output_dir              : {out_dir}")
    print(f"  matching_rule           : {sub['matching_rule']}")
    print(f"  tolerance_m             : {sub['tolerance_m']}")
    print(f"  sec_agreement_min       : {sub['sec_agreement_min']}")
    print(f"  caliper_severity_filter : {sub['caliper_severity_filter']}")
    print(f"  best_match_priority     : {sub['best_match_priority']}")
    print(f"  severity_weights        : {sub['severity_weights']}")
    print(f"  unmatched_min_severity  : {sub['unmatched_zones_min_severity']}")
    print(f"  run_tag                 : {sub['run_tag']}")
    print("=" * 72)
    print()

    # ── Load inputs ──
    if not sec_path.exists():
        print(f"ERROR: SEC clusters file not found: {sec_path}", file=sys.stderr)
        return 1
    if not cal_path.exists():
        print(f"ERROR: caliper zones file not found: {cal_path}", file=sys.stderr)
        return 1

    sec_clusters = load_sec_clusters(sec_path)
    caliper_zones = load_caliper_zones(cal_path)

    print(f"  SEC clusters loaded     : {len(sec_clusters)}  "
          f"(wells: {sorted(sec_clusters['well_id'].unique())})")
    print(f"  caliper zones loaded    : {len(caliper_zones)}  "
          f"(wells: {sorted(caliper_zones['well_id'].unique())})")
    n_kept = int((sec_clusters["agreement"] >= sub["sec_agreement_min"]).sum())
    print(f"  clusters after filter   : {n_kept} "
          f"(agreement >= {sub['sec_agreement_min']})")
    print()

    # ── Run ──
    result = compute_convergence(sec_clusters, caliper_zones, config=sub)

    # ── Write outputs ──
    out_clusters = out_dir / "cluster_matches.csv"
    out_summary = out_dir / "well_summary.csv"
    out_unmatched = out_dir / "unmatched_caliper_zones.csv"

    result.cluster_matches.to_csv(out_clusters, index=False)
    result.well_summary.to_csv(out_summary, index=False)
    result.unmatched_zones.to_csv(out_unmatched, index=False)

    print(f"  wrote: {out_clusters}")
    print(f"  wrote: {out_summary}")
    print(f"  wrote: {out_unmatched}")

    # ── Print summary table ──
    print()
    print("─" * 72)
    print(" Per-well summary")
    print("─" * 72)
    cols = [
        "well_id", "n_clusters_analyzed", "n_converging",
        "fraction_converging", "n_with_severe_match",
        "n_with_moderate_match", "n_with_mild_match",
        "max_convergence_score",
    ]
    print(result.well_summary[cols].to_string(index=False))
    print()

    # ── Ledger entry (one aggregated row, well_id='ALL') ──
    if not args.no_ledger:
        params = params_for_run_ledger(cfg, "convergence_sec_caliper")
        try:
            # We use register_run for its hashing/signature/timestamp
            # plumbing, but our outputs are aggregated multi-well CSVs
            # with stable names — not the auto-built per-well filename.
            # So we point output_file at the output directory before exit.
            with register_run(
                stage="convergence_sec_caliper",
                well_id="ALL",
                date=campaign,    # campaign is the natural date-like key here
                input_file=f"{sec_path} + {cal_path}",
                params=params,
                output_dir=out_dir,
                extension="csv",
            ) as run:
                run.note = (
                    f"SEC↔caliper convergence (Idea 3). "
                    f"agreement_min={sub['sec_agreement_min']}, "
                    f"matching_rule={sub['matching_rule']}, "
                    f"tol={sub['tolerance_m']}m. "
                    f"Outputs in folder: cluster_matches.csv, "
                    f"well_summary.csv, unmatched_caliper_zones.csv."
                )
                # Override output_file so the ledger row points at the
                # folder (where the three CSVs live), not at the
                # auto-generated per-well filename which we don't use.
                run.output_file = str(out_dir)
            print(f"  ledger entry recorded: stage='convergence_sec_caliper', "
                  f"well_id='ALL'")
        except Exception as exc:
            # Don't fail the whole run for a ledger hiccup.
            print(f"  WARNING: could not write ledger entry: {exc}",
                  file=sys.stderr)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
