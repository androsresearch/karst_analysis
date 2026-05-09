"""Throwaway: print a side-by-side summary of all trials in every JSON.

For each .json in data/breakpoints/<campaign>/, prints:
  - well_id, method, trial name
  - BIC of the model at the chosen N
  - the breakpoint X positions of that model

This lets you decide quickly whether the 3 trials per (well, method) are
consistent (you can pick trial_1 by default) or wildly different (you
need to choose per case).

Run from the repo root:
    uv run python scripts/diagnostics/inspect_trials.py --campaign 2022_02 --n 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _df_from_records(records):
    """The JSON stores each trial as a list-of-records dict. Reconstruct."""
    return pd.DataFrame(records)


def _bp_positions_at_n(df: pd.DataFrame, n: int) -> list[float]:
    """Extract the breakpoint X positions at row n_breakpoints == n."""
    row = df[df["n_breakpoints"] == n]
    if len(row) == 0 or not row["converged"].iloc[0]:
        return []
    est = row["estimates"].iloc[0]
    if not isinstance(est, dict):
        return []
    # Estimates dict has keys 'breakpoint1', 'breakpoint2', ... each with
    # 'estimate', 'confidence_interval', etc.
    bps = []
    for k in sorted(est.keys()):
        if k.startswith("breakpoint"):
            v = est[k]
            if isinstance(v, dict) and "estimate" in v:
                bps.append(float(v["estimate"]))
    return bps


def _bic_at_n(df: pd.DataFrame, n: int) -> float | None:
    row = df[df["n_breakpoints"] == n]
    if len(row) == 0:
        return None
    bic = row["bic"].iloc[0]
    return float(bic) if pd.notna(bic) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--n", type=int, default=15,
                   help="Inspect the model at this N (default 15).")
    p.add_argument("--bp-dir", type=Path, default=None)
    args = p.parse_args()

    bp_dir = args.bp_dir or Path(f"data/breakpoints/{args.campaign}")
    if not bp_dir.exists():
        raise SystemExit(f"bp-dir does not exist: {bp_dir}")

    rows = []
    for jp in sorted(bp_dir.glob("*__bp-*.json")):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # filename → well_id, method
        stem = jp.stem
        well_date_part, method_sig = stem.split("__", 1)
        well_id, date = well_date_part.rsplit("_", 1)
        method = "savgol" if "savgol" in method_sig else "lowess"

        for trial_name, trial_data in data.items():
            df_trial = _df_from_records(trial_data["df"])
            bic = _bic_at_n(df_trial, args.n)
            bps = _bp_positions_at_n(df_trial, args.n)

            rows.append({
                "well": well_id,
                "method": method,
                "trial": trial_name,
                "bic_at_N": round(bic, 1) if bic is not None else None,
                "n_bps_found": len(bps),
                "bp_first": round(bps[0], 2) if bps else None,
                "bp_last": round(bps[-1], 2) if bps else None,
                "bps": [round(b, 2) for b in bps],
            })

    if not rows:
        print("No JSONs found.")
        return

    summary = pd.DataFrame(rows)
    print(f"\nTrials at N={args.n}, campaign={args.campaign}\n")
    # Sort by well, method, trial
    summary = summary.sort_values(["well", "method", "trial"]).reset_index(drop=True)

    # Compact display: drop the bps list (too long for a table)
    cols_short = ["well", "method", "trial", "bic_at_N",
                  "n_bps_found", "bp_first", "bp_last"]
    print(summary[cols_short].to_string(index=False))

    # Detailed: which trial has best BIC per (well, method) and how
    # different are the BPs across trials
    print()
    print("BIC ranking per (well, method):")
    for (well, method), grp in summary.groupby(["well", "method"]):
        ranked = grp.sort_values("bic_at_N").reset_index(drop=True)
        if len(ranked) == 0 or ranked["bic_at_N"].isna().all():
            continue
        bics = ranked["bic_at_N"].dropna()
        spread = bics.max() - bics.min() if len(bics) > 1 else 0
        print(f"  {well} {method:7s}: best={ranked['trial'].iloc[0]} "
              f"(BIC={bics.min():.0f}), worst BIC={bics.max():.0f}, "
              f"spread={spread:.0f}")

    print()
    print("BP positions per trial (full list):")
    for (well, method), grp in summary.groupby(["well", "method"]):
        print(f"  {well} {method}:")
        for _, r in grp.iterrows():
            print(f"    {r['trial']}  BIC={r['bic_at_N']}  bps={r['bps']}")


if __name__ == "__main__":
    main()
