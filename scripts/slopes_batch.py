"""Batch: compute chord slopes and identify mixing-zone boundaries.

This script reads a YAML "jobs file" that explicitly lists which
(well, method, trial, N) combinations to process. It does NOT
auto-discover tasks; the user picks each combination after inspecting
the breakpoint figures (see also
``scripts/diagnostics/render_all_trials.py``).

Jobs file format
----------------
    # config/slopes_jobs_2022_02.yml
    campaign: 2022_02
    jobs:
      - well: LRS70D
        method: lowess
        trial: trial_1
        n: 15
      - well: AW5D
        method: savgol
        trial: trial_3
        n: 15
      # ...

For every job this script produces:
    1. A CSV with the slopes table:
       data/slopes/<campaign>/{well_id}_{date}__slopes-{method}-N{n}-t{idx}.csv
    2. Two PNG figures (log10 and linear x-axis) under
       results/figures/slopes/<campaign>/{well_id}/

Usage
-----
    uv run python scripts/slopes_batch.py --jobs config/slopes_jobs_2022_02.yml

Override paths or skip plots:
    uv run python scripts/slopes_batch.py --jobs config/slopes_jobs.yml --no-plot
    uv run python scripts/slopes_batch.py --jobs config/slopes_jobs.yml --no-linear
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from karst_analysis.sec.breakpoints import (
    extract_breakpoints,
    get_breakpoint_data,
    rebuild_model,
)
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.slopes import compute_slopes
from karst_analysis.sec.viz import plot_slopes_overlay


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("slopes_batch")


# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    """One unit of work: process this exact (well, method, trial, n)."""
    well: str
    method: str
    trial: str
    n: int
    bot_mz_sec_threshold: Optional[float] = None  # None → use script default


def _parse_well_id(well_id: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)([DSO])$", well_id)
    if m is None:
        raise ValueError(f"Cannot split well_id '{well_id}'")
    return m.group(1), m.group(2)


def _trial_index(trial_name: str) -> int:
    m = re.search(r"(\d+)$", trial_name)
    return int(m.group(1)) if m else 1


def _normalise_campaign(value) -> str:
    """Coerce a campaign value to canonical 'YYYY_MM' string.

    PyYAML interprets unquoted '2022_02' as the integer 202202 (Python
    allows underscores as digit separators). To stay backwards-friendly
    we accept both: if we receive a 6-digit int, we reformat it as
    'YYYY_MM' and warn the user. Strings pass through verbatim.
    """
    if isinstance(value, int):
        s = str(value)
        if len(s) == 6 and s[:4].isdigit() and s[4:].isdigit():
            recovered = f"{s[:4]}_{s[4:]}"
            logger.warning(
                f"campaign was parsed as integer {value} (the YAML value "
                f"likely lacked quotes). Reconstructed as '{recovered}'. "
                f"Consider quoting it in the YAML: campaign: \"{recovered}\""
            )
            return recovered
        return s
    if isinstance(value, str):
        return value
    raise ValueError(
        f"campaign must be a string (e.g. '2022_02'); got {type(value).__name__}: {value!r}"
    )


def _load_jobs_file(path: Path) -> tuple[str, Optional[float], list[Job]]:
    """Parse a YAML jobs file.

    Returns
    -------
    (campaign, default_threshold, jobs)
        - campaign: canonical 'YYYY_MM' string
        - default_threshold: optional global default for
          bot_mz_sec_threshold; None if not specified at YAML root.
        - jobs: list of Job dataclasses; per-job threshold is on
          ``bot_mz_sec_threshold`` if specified, else None.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must be a YAML mapping with 'campaign' and 'jobs'.")
    raw_campaign = cfg.get("campaign")
    if raw_campaign is None:
        raise ValueError(f"{path}: missing 'campaign' key.")
    campaign = _normalise_campaign(raw_campaign)

    default_threshold = cfg.get("bot_mz_sec_threshold")
    if default_threshold is not None:
        default_threshold = float(default_threshold)

    jobs_raw = cfg.get("jobs", [])
    if not jobs_raw:
        raise ValueError(f"{path}: 'jobs' list is empty.")
    jobs: list[Job] = []
    for i, j in enumerate(jobs_raw, start=1):
        for k in ("well", "method", "trial", "n"):
            if k not in j:
                raise ValueError(f"{path} job #{i}: missing key '{k}'.")
        if j["method"] not in ("savgol", "lowess"):
            raise ValueError(
                f"{path} job #{i}: method must be 'savgol' or 'lowess'."
            )
        per_job_threshold = j.get("bot_mz_sec_threshold")
        if per_job_threshold is not None:
            per_job_threshold = float(per_job_threshold)
        jobs.append(Job(
            well=str(j["well"]),
            method=str(j["method"]),
            trial=str(j["trial"]),
            n=int(j["n"]),
            bot_mz_sec_threshold=per_job_threshold,
        ))
    return campaign, default_threshold, jobs


def _find_raw_csv(raw_dir: Path, well: str, date: str) -> Optional[Path]:
    site, well_type = _parse_well_id(well)
    cands = list(raw_dir.glob(f"{site}_{well_type}_*_{date}.csv"))
    if not cands:
        cands = list(raw_dir.glob(f"{site}{well_type}_*_{date}.csv"))
    return cands[0] if cands else None


def _find_processed_csv(proc_dir: Path, well: str, date: str) -> Optional[Path]:
    if not proc_dir.exists():
        return None
    cands = sorted(proc_dir.glob(f"{well}_{date}__*.csv"))
    return cands[-1] if cands else None


def _find_breakpoint_json(bp_dir: Path, well: str, method: str) -> Optional[Path]:
    """Find the breakpoint JSON for (well, method) regardless of date/N."""
    cands = sorted(bp_dir.glob(f"{well}_*__bp-{method}-*.json"))
    return cands[0] if cands else None


def _date_from_json_name(jp: Path) -> str:
    """Extract YYYYMMDD from '<well>_<date>__bp-<method>-...'."""
    stem = jp.stem
    well_date_part, _ = stem.split("__", 1)
    _, date = well_date_part.rsplit("_", 1)
    return date


# ─────────────────────────────────────────────────────────────────────────
def _process_job(
    job: Job,
    *,
    bp_dir: Path,
    proc_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    do_plot: bool,
    do_linear: bool,
    default_threshold: Optional[float] = None,
) -> dict:
    """Execute one job. Returns a summary row.

    Threshold resolution:
    - If the job specifies its own ``bot_mz_sec_threshold``, that wins.
    - Else if the YAML root specified a default, use that.
    - Else fall back to compute_slopes' built-in default (40 000).
    """
    t0 = time.time()

    # Locate JSON
    jp = _find_breakpoint_json(bp_dir, job.well, job.method)
    if jp is None:
        raise RuntimeError(
            f"No breakpoint JSON for {job.well} {job.method} in {bp_dir}"
        )
    date = _date_from_json_name(jp)

    # Load JSON, validate trial
    with open(jp, "r", encoding="utf-8") as f:
        all_trials = json.load(f)
    if job.trial not in all_trials:
        raise RuntimeError(
            f"Trial '{job.trial}' not in {jp.name}. "
            f"Available: {list(all_trials.keys())}"
        )
    trial_data = all_trials[job.trial]
    # NB: trial_data["df"] is a dict-of-dicts (keys are strings of N
    # values). Do NOT wrap in pd.DataFrame — get_breakpoint_data
    # expects the raw dict format that comes straight from the JSON.

    # Validate that this N converged in this trial
    bp_params = get_breakpoint_data(trial_data["df"], job.n)
    if isinstance(bp_params, str):
        raise RuntimeError(
            f"{job.well} {job.method} {job.trial} N={job.n}: {bp_params}"
        )

    # Load smoothed CSV
    method_proc_dir = proc_dir / job.method
    csv_path = _find_processed_csv(method_proc_dir, job.well, date)
    if csv_path is None:
        raise RuntimeError(
            f"No smoothed CSV for {job.well} ({job.method}) in {method_proc_dir}"
        )
    df_sm = pd.read_csv(csv_path)
    mask = df_sm["log10_sec_uS_cm"].notna()
    xx = df_sm.loc[mask, "depth_m"].to_numpy()
    yy = df_sm.loc[mask, "log10_sec_uS_cm"].to_numpy()

    # Rebuild model and extract ordered breakpoints
    model = rebuild_model(xx, yy, bp_params)
    bp_df = extract_breakpoints(model)

    # Compute slopes + curvature flags
    # Resolve threshold: per-job > YAML default > compute_slopes default
    if job.bot_mz_sec_threshold is not None:
        threshold = job.bot_mz_sec_threshold
    elif default_threshold is not None:
        threshold = default_threshold
    else:
        threshold = 40_000.0  # matches compute_slopes default
    slopes = compute_slopes(bp_df, bot_mz_sec_threshold=threshold)

    # Persist CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_idx = _trial_index(job.trial)
    csv_name = (
        f"{job.well}_{date}__"
        f"slopes-{job.method}-N{job.n}-t{trial_idx}.csv"
    )
    csv_out = out_dir / csv_name
    slopes.to_csv(csv_out, index=False)

    # Plots
    fig_paths: list[Path] = []
    if do_plot:
        raw_csv = _find_raw_csv(raw_dir, job.well, date)
        if raw_csv is None:
            logger.warning(f"  no raw CSV for {job.well}_{date}; skipping plots")
        else:
            df_raw = load_ysi_csv(raw_csv, standardise=True)
            z_raw  = df_raw["depth_m"].to_numpy()
            ec_raw = df_raw["sec_uS_cm"].to_numpy()
            z_sm   = df_sm["depth_m"].to_numpy()
            ec_sm  = df_sm["sec_uS_cm"].to_numpy()

            well_fig_dir = fig_dir / job.well
            well_fig_dir.mkdir(parents=True, exist_ok=True)
            title_base = (
                f"{job.well} · {date} · {job.method.upper()} · "
                f"N={job.n} · {job.trial}"
            )

            # log10
            try:
                p_log = plot_slopes_overlay(
                    z_raw=z_raw, EC_raw=ec_raw,
                    z_smooth=z_sm, EC_smooth=ec_sm,
                    slopes_df=slopes,
                    output_path=well_fig_dir / (
                        f"{job.well}_{date}_"
                        f"slopes-{job.method}-N{job.n}-t{trial_idx}_log10.png"
                    ),
                    axis_scale="log10",
                    title=f"{title_base}  —  log₁₀(SEC)",
                    method_label=job.method.upper(),
                )
                fig_paths.append(p_log)
            except Exception as e:
                logger.warning(f"  log10 plot failed: {e}")

            # linear
            if do_linear:
                try:
                    p_lin = plot_slopes_overlay(
                        z_raw=z_raw, EC_raw=ec_raw,
                        z_smooth=z_sm, EC_smooth=ec_sm,
                        slopes_df=slopes,
                        output_path=well_fig_dir / (
                            f"{job.well}_{date}_"
                            f"slopes-{job.method}-N{job.n}-t{trial_idx}_linear.png"
                        ),
                        axis_scale="linear",
                        title=f"{title_base}  —  linear SEC",
                        method_label=job.method.upper(),
                    )
                    fig_paths.append(p_lin)
                except Exception as e:
                    logger.warning(f"  linear plot failed: {e}")

    elapsed = time.time() - t0

    # Mixing-zone summary
    top_row = slopes.loc[slopes["is_top_of_mixing"]]
    bot_row = slopes.loc[slopes["is_bottom_of_mixing"]]
    top_z = float(top_row["depth_top"].iloc[0]) if len(top_row) else None
    bot_z = float(bot_row["depth_top"].iloc[0]) if len(bot_row) else None

    # TODO (v15.1): register this run in results/runs.csv via the
    # run_ledger context manager, the way breakpoints_batch.py does.
    # Deferred until we settle on the schema for slopes runs.

    return {
        "well":          job.well,
        "date":          date,
        "method":        job.method,
        "trial":         job.trial,
        "n":             job.n,
        "threshold":     threshold,
        "csv":           csv_out.name,
        "n_figures":     len(fig_paths),
        "top_mz_depth":  top_z,
        "bot_mz_depth":  bot_z,
        "elapsed_s":     elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True,
                   help="Path to YAML jobs file.")
    p.add_argument("--bp-dir", type=Path, default=None)
    p.add_argument("--proc-dir", type=Path, default=None)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-linear", action="store_true")
    args = p.parse_args(argv)

    if not args.jobs.exists():
        logger.error(f"Jobs file does not exist: {args.jobs}")
        return 1

    campaign, default_threshold, jobs = _load_jobs_file(args.jobs)

    bp_dir   = args.bp_dir   or Path(f"data/breakpoints/{campaign}")
    proc_dir = args.proc_dir or Path(f"data/processed/sec/{campaign}")
    raw_dir  = args.raw_dir  or Path(f"data/raw/sec/{campaign}")
    out_dir  = args.out_dir  or Path(f"data/slopes/{campaign}")
    fig_dir  = args.fig_dir  or Path(f"results/figures/slopes/{campaign}")

    print("=" * 72)
    print(" SLOPES BATCH (jobs-driven)")
    print("=" * 72)
    print(f"  jobs file : {args.jobs}")
    print(f"  campaign  : {campaign}")
    print(f"  jobs      : {len(jobs)}")
    print(f"  bp-dir    : {bp_dir}")
    print(f"  proc-dir  : {proc_dir}")
    print(f"  out-dir   : {out_dir}")
    print(f"  fig-dir   : {fig_dir}")
    threshold_display = (
        f"{default_threshold} µS/cm" if default_threshold is not None
        else "40000 µS/cm (compute_slopes default)"
    )
    print(f"  threshold : {threshold_display}  (BOT MZ; jobs may override)")
    print(f"  plot      : {'NO' if args.no_plot else 'YES'}")
    if not args.no_plot:
        print(f"  linear    : {'NO' if args.no_linear else 'YES'}")
    print("=" * 72)

    rows: list[dict] = []
    failed: list[tuple[Job, str]] = []
    t_start = time.time()

    for i, job in enumerate(jobs, start=1):
        logger.info(
            f"[{i}/{len(jobs)}] {job.well} {job.method} {job.trial} N={job.n}"
        )
        try:
            row = _process_job(
                job,
                bp_dir=bp_dir, proc_dir=proc_dir, raw_dir=raw_dir,
                out_dir=out_dir, fig_dir=fig_dir,
                do_plot=not args.no_plot,
                do_linear=not args.no_linear,
                default_threshold=default_threshold,
            )
            rows.append(row)
            top_str = (
                f"TOP={row['top_mz_depth']:.2f} m"
                if row['top_mz_depth'] is not None else "TOP=—"
            )
            bot_str = (
                f"BOT={row['bot_mz_depth']:.2f} m"
                if row['bot_mz_depth'] is not None else "BOT unmarked"
            )
            logger.info(
                f"  ✓ {row['elapsed_s']:.1f} s → {row['csv']}  "
                f"({top_str}  {bot_str}, {row['n_figures']} fig)"
            )
        except Exception as e:
            failed.append((job, str(e)))
            logger.error(f"  ✗ FAILED: {e}")

    total_min = (time.time() - t_start) / 60.0
    print()
    print("=" * 72)
    print(f" SUMMARY — {len(rows)} jobs OK, {len(failed)} failed, "
          f"in {total_min:.1f} min")
    print("=" * 72)
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    if failed:
        print()
        print("Failed jobs:")
        for j, msg in failed:
            print(f"  - {j.well} {j.method} {j.trial} N={j.n}  →  {msg}")

    return 0 if rows and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
