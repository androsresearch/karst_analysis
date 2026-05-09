"""One-time utility: flip the sign of the ``Depth [m]`` column in the
master concatenated caliper CSV so that depths are positive (BGL).

Background
----------
The original LAS files reported depth as positive metres below ground
level (BGL convention). The ``concatenate_caliper_all.csv`` master
file currently in this repo was produced by an earlier pipeline that
multiplied ``Depth [m]`` by -1 (an "elevation" convention). v5.2 of
``karst_analysis`` standardises on BGL positive throughout, so the
master file must be reverted to its original sign.

This script:
    1. Reads ``data/raw/caliper/concatenate_caliper_all.csv``.
    2. Verifies the depth column currently has negative values (so we
       only flip if it actually needs flipping — running twice does
       nothing).
    3. Saves a backup at
       ``data/raw/caliper/concatenate_caliper_all_backup_negative.csv``.
    4. Multiplies ``Depth [m]`` by -1 in place and writes the original
       file back.

Usage
-----
    uv run python scripts/fix_caliper_master_signs.py

After running, the SEC + caliper + videolog pipelines all share the
"depth below ground level, positive" convention.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


MASTER_CSV = Path("data/raw/caliper/concatenate_caliper_all.csv")
BACKUP_CSV = Path("data/raw/caliper/concatenate_caliper_all_backup_negative.csv")


def main() -> int:
    if not MASTER_CSV.exists():
        print(f"ERROR: master CSV not found: {MASTER_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(MASTER_CSV)
    if "Depth [m]" not in df.columns:
        print(f"ERROR: 'Depth [m]' column not in {MASTER_CSV}", file=sys.stderr)
        return 1

    z_min, z_max = df["Depth [m]"].min(), df["Depth [m]"].max()
    print(f"Current depth range: [{z_min:.3f}, {z_max:.3f}] m")

    # Detect convention: positive depth means BGL (already correct);
    # negative depth means elevation (needs flipping).
    if z_min >= 0:
        print("✓ Master CSV already in BGL-positive convention. Nothing to do.")
        return 0
    if z_max > 0:
        print(f"WARNING: depth column has both positive and negative values "
              f"(min={z_min}, max={z_max}). This is unexpected; refusing to "
              f"flip blindly. Inspect the file manually.", file=sys.stderr)
        return 2

    # All depths negative → flip
    if not BACKUP_CSV.exists():
        shutil.copy2(MASTER_CSV, BACKUP_CSV)
        print(f"✓ Backup saved: {BACKUP_CSV}")
    else:
        print(f"  (backup already exists at {BACKUP_CSV}, not overwriting)")

    df["Depth [m]"] = -df["Depth [m]"]
    df.to_csv(MASTER_CSV, index=False)
    z_min_new, z_max_new = df["Depth [m]"].min(), df["Depth [m]"].max()
    print(f"✓ Flipped. New depth range: [{z_min_new:.3f}, {z_max_new:.3f}] m")
    print(f"✓ Wrote: {MASTER_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
