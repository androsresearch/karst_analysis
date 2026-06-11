"""Build a long-format CSV summarising the canonical SEC breakpoints.

For each (well, smoothing_method, trial, N) job listed in a
`config/slopes_jobs_<campaign>.yml` file, this script reads the
breakpoints JSON and the matching slopes CSV, looks up the well's
vadose-zone thickness from `data/metadata/wells.csv`, and emits one
row per breakpoint with both water-table and BGL depths plus the
TOP MZ / BOT MZ flags.

The output is the human-friendly cross-check table that the
machine-readable JSONs and slopes CSVs do not provide: everything a
reader needs to verify a breakpoint depth at a glance lives in one
file.

Output schema (one row per breakpoint):

  well_id           - e.g. "LRS70D"
  campaign          - e.g. "2022_02"
  date              - cast date in YYYY-MM-DD, e.g. "2022-01-31"
  method            - smoothing method (lowess / savgol)
  trial             - "trial_1" / "trial_2" / "trial_3"
  n_breakpoints     - 15 in the current canonical jobs
  bp_idx            - 1-based breakpoint index
  depth_wt_m        - depth below water table (native frame of the
                      JSON estimate)
  vadose_m          - vadose-zone thickness for the well, from
                      wells.csv (campaign-matched)
  depth_bgl_m       - depth below ground level = depth_wt_m + vadose_m
  is_top_mz         - True iff this BP is flagged TOP of mixing zone
                      in the slopes CSV
  is_bot_mz         - True iff this BP is flagged BOT of mixing zone

Usage:

    uv run python scripts/build_sec_breakpoints_summary.py \\
        --jobs config/slopes_jobs_2022_02.yml

The output goes to `results/sec_breakpoints_summary_<campaign>.csv`
by default; override with `--out-csv`.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# These imports rely on `karst_analysis` being installed in the env
# (uv pip install -e .). Same as the rest of the scripts in this dir.
from karst_analysis.corrections import get_vadose_thickness, load_well_metadata
from karst_analysis.sec.jobs_io import load_jobs_file as _load_jobs_file, trial_index as _trial_index


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_sec_breakpoints_summary")


# ─────────────────────────────────────────────────────────────────────
def _find_breakpoint_json(bp_dir: Path, well: str, method: str) -> Optional[Path]:
    cands = sorted(bp_dir.glob(f"{well}_*__bp-{method}-*.json"))
    return cands[0] if cands else None


def _find_slopes_csv(slopes_dir: Path, well: str, method: str, n: int, t_idx: int) -> Optional[Path]:
    cands = sorted(slopes_dir.glob(f"{well}_*__slopes-{method}-N{n}-t{t_idx}.csv"))
    return cands[0] if cands else None


def _date_from_json_name(jp: Path) -> str:
    """'BW3D_20220202__bp-lowess-...json' -> '2022-02-02'."""
    stem = jp.stem
    well_date_part, _ = stem.split("__", 1)
    _, yyyymmdd = well_date_part.rsplit("_", 1)
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _process_job(job, *, bp_dir: Path, slopes_dir: Path, well_metadata: pd.DataFrame,
                 campaign: str) -> list[dict]:
    """Build the rows for one canonical (well, method, trial, N) job."""
    # JSON
    jp = _find_breakpoint_json(bp_dir, job.well, job.method)
    if jp is None:
        raise RuntimeError(f"No breakpoint JSON for {job.well} {job.method} in {bp_dir}")
    date_iso = _date_from_json_name(jp)

    with open(jp, "r", encoding="utf-8") as f:
        all_trials = json.load(f)
    if job.trial not in all_trials:
        raise RuntimeError(
            f"Trial {job.trial!r} not in {jp.name}. "
            f"Available: {list(all_trials.keys())}"
        )

    n_str = str(job.n)
    estimates = all_trials[job.trial]["df"]["estimates"]
    if n_str not in estimates:
        raise RuntimeError(
            f"n_breakpoints={n_str} not in {jp.name}/{job.trial}/df/estimates"
        )

    bps_wt = [estimates[n_str][f"breakpoint{i}"]["estimate"]
              for i in range(1, job.n + 1)]

    # Slopes CSV
    t_idx = _trial_index(job.trial)
    sp = _find_slopes_csv(slopes_dir, job.well, job.method, job.n, t_idx)
    if sp is None:
        raise RuntimeError(
            f"No slopes CSV for {job.well} {job.method} N={job.n} t{t_idx} in {slopes_dir}"
        )
    slopes_df = pd.read_csv(sp)

    # Sanity: same series in WT.
    reconstructed_wt = (
        list(slopes_df["depth_top"].to_numpy())
        + [float(slopes_df["depth_bottom"].iloc[-1])]
    )
    if len(reconstructed_wt) != len(bps_wt):
        raise RuntimeError(
            f"{job.well} {job.method} {job.trial}: slopes CSV implies "
            f"{len(reconstructed_wt)} BPs; JSON has {len(bps_wt)}"
        )
    import numpy as np
    max_disagr = float(np.max(np.abs(np.array(reconstructed_wt) - np.array(bps_wt))))
    if max_disagr > 1e-3:
        raise RuntimeError(
            f"{job.well} {job.method} {job.trial}: slopes CSV vs JSON disagree "
            f"by up to {max_disagr:.4f} m (WT frame)"
        )

    # MZ flags per BP (positions 0..N-2; the last BP carries no flag).
    is_top    = slopes_df["is_top_of_mixing"].to_numpy(dtype=bool)
    is_bottom = slopes_df["is_bottom_of_mixing"].to_numpy(dtype=bool)

    # Vadose lookup.
    vadose_m = get_vadose_thickness(job.well, metadata=well_metadata)

    rows: list[dict] = []
    for i, bp_wt in enumerate(bps_wt):  # i is 0-based
        bp_idx = i + 1
        rows.append({
            "well_id":       job.well,
            "campaign":      campaign,
            "date":          date_iso,
            "method":        job.method,
            "trial":         job.trial,
            "n_breakpoints": job.n,
            "bp_idx":        bp_idx,
            "depth_wt_m":    round(float(bp_wt), 4),
            "vadose_m":      round(float(vadose_m), 4),
            "depth_bgl_m":   round(float(bp_wt + vadose_m), 4),
            "is_top_mz":     bool(is_top[i])    if i < len(is_top)    else False,
            "is_bot_mz":     bool(is_bottom[i]) if i < len(is_bottom) else False,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True,
                   help="YAML jobs file (one campaign).")
    p.add_argument("--bp-dir", type=Path, default=None)
    p.add_argument("--slopes-dir", type=Path, default=None)
    p.add_argument("--wells-csv", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.jobs.exists():
        logger.error(f"Jobs file does not exist: {args.jobs}")
        return 1

    campaign, _default_bot_threshold, _default_top_threshold, jobs = _load_jobs_file(args.jobs)

    bp_dir     = args.bp_dir     or Path(f"data/breakpoints/{campaign}")
    slopes_dir = args.slopes_dir or Path(f"data/slopes/{campaign}")
    wells_csv  = args.wells_csv  or Path("data/metadata/wells.csv")
    out_csv    = args.out_csv    or Path(f"results/sec_breakpoints_summary_{campaign}.csv")

    well_metadata = load_well_metadata(wells_csv)

    print("=" * 72)
    print(" SEC BREAKPOINTS SUMMARY")
    print("=" * 72)
    print(f"  jobs       : {args.jobs}  ({len(jobs)} jobs)")
    print(f"  bp-dir     : {bp_dir}")
    print(f"  slopes-dir : {slopes_dir}")
    print(f"  wells-csv  : {wells_csv}")
    print(f"  out-csv    : {out_csv}")
    print("=" * 72)

    all_rows: list[dict] = []
    failed: list[tuple] = []
    for i, job in enumerate(jobs, start=1):
        logger.info(f"[{i}/{len(jobs)}] {job.well} {job.method} {job.trial} N={job.n}")
        try:
            rows = _process_job(
                job,
                bp_dir=bp_dir, slopes_dir=slopes_dir,
                well_metadata=well_metadata, campaign=campaign,
            )
            all_rows.extend(rows)
            top_rows = [r for r in rows if r["is_top_mz"]]
            bot_rows = [r for r in rows if r["is_bot_mz"]]
            top_z = (f"TOP={top_rows[0]['depth_bgl_m']:.2f} m BGL"
                     if top_rows else "TOP=—")
            bot_z = (f"BOT={bot_rows[0]['depth_bgl_m']:.2f} m BGL"
                     if bot_rows else "BOT=—")
            logger.info(f"  ✓ {len(rows)} rows; {top_z}, {bot_z}")
        except Exception as e:
            failed.append((job, str(e)))
            logger.error(f"  ✗ FAILED: {e}")

    if not all_rows:
        logger.error("No rows produced.")
        return 1

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["well_id", "method", "trial", "bp_idx"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print("=" * 72)
    print(f" {len(df)} rows over {df['well_id'].nunique()} well(s).")
    print(f" Written: {out_csv}")
    print("=" * 72)

    # One-line MZ summary per well, for the console.
    print()
    print(" MZ summary per well (BGL):")
    summary = (
        df.loc[df["is_top_mz"] | df["is_bot_mz"]]
          .pivot_table(
              index=["well_id"],
              columns=[df["is_top_mz"].map({True: "TOP_MZ", False: ""}).where(df["is_top_mz"], "BOT_MZ")],
              values="depth_bgl_m",
              aggfunc="first",
          )
    )
    # Simpler: explicit print loop.
    for wid in sorted(df["well_id"].unique()):
        sub = df[df["well_id"] == wid]
        tops = sub.loc[sub["is_top_mz"], "depth_bgl_m"].tolist()
        bots = sub.loc[sub["is_bot_mz"], "depth_bgl_m"].tolist()
        top_s = f"{tops[0]:.2f} m" if tops else "—"
        bot_s = f"{bots[0]:.2f} m" if bots else "—"
        print(f"   {wid:<7}  TOP MZ = {top_s:>10}    BOT MZ = {bot_s:>10}")

    if failed:
        print()
        print("Failed jobs:")
        for j, msg in failed:
            print(f"  - {j.well} {j.method} {j.trial} N={j.n}  ->  {msg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
