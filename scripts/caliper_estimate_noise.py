"""Estimate caliper instrumental noise (AW5O vs AW5D) and write a JSON report.

The output JSON has the schema documented in
``karst_analysis.caliper.noise.estimate_noise_aw5o_vs_aw5d`` and is the
input to the breakouts pipeline.

Usage
-----
    uv run python scripts/caliper_estimate_noise.py
    uv run python scripts/caliper_estimate_noise.py --master data/raw/caliper/concatenate_caliper_all.csv
    uv run python scripts/caliper_estimate_noise.py --output results/noise_comparison.json

Default I/O paths
-----------------
    --master  : data/raw/caliper/concatenate_caliper_all.csv
    --output  : data/processed/caliper/noise_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from karst_analysis.caliper.io import load_master_caliper, DEFAULT_MASTER_CSV
from karst_analysis.caliper.noise import estimate_noise_aw5o_vs_aw5d


DEFAULT_OUTPUT = Path("data/processed/caliper/noise_comparison.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=str(DEFAULT_MASTER_CSV),
                   help=f"Master caliper CSV (default: {DEFAULT_MASTER_CSV}).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the summary printout.")
    args = p.parse_args()

    master_path = Path(args.master)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 72)
        print(" CALIPER NOISE ESTIMATE — AW5O vs AW5D")
        print("=" * 72)
        print(f"  master : {master_path}")
        print(f"  output : {output_path}")
        print("=" * 72)

    df = load_master_caliper(master_path)
    report = estimate_noise_aw5o_vs_aw5d(df)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    if not args.quiet:
        o = report["AW5O"]
        d = report["AW5D"]
        c = report["comparison"]
        print()
        print(f"  AW5O ({o['drilling_method']}, {o['auger_in']:.0f}\" auger):")
        print(f"     interval     : {o['well_interval']} m")
        print(f"     n samples    : {o['n']}")
        print(f"     sigma_std    : {o['sigma_std_cm']:.4f} cm")
        print(f"     sigma_MAD    : {o['sigma_MAD_cm']:.4f} cm  ← used downstream")
        print(f"     lag-1 autoc. : {o['lag1_autocorr']:+.3f}")
        print()
        print(f"  AW5D ({d['drilling_method']}, {d['auger_in']:.0f}\" auger):")
        print(f"     interval     : {d['well_interval']} m")
        print(f"     n samples    : {d['n']}")
        print(f"     sigma_std    : {d['sigma_std_cm']:.4f} cm")
        print(f"     sigma_MAD    : {d['sigma_MAD_cm']:.4f} cm")
        print(f"     lag-1 autoc. : {d['lag1_autocorr']:+.3f}")
        print()
        print("  Variance decomposition (Gaussian, independent):")
        print(f"     sigma_drilling (from std) = {c['sigma_drilling_from_std_cm']:.4f} cm")
        print(f"     sigma_drilling (from MAD) = {c['sigma_drilling_from_MAD_cm']:.4f} cm")
        print()
        print(f"✓ Saved {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
