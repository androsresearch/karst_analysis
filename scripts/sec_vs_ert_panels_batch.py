"""Batch: produce all SEC vs ERT 1D comparison panels.

Reads three configuration sources:

  1. ``config/slopes_jobs_<campaign>.yml``  -- which (well, method,
     trial, n) was chosen for SEC. This is the SAME file that
     ``scripts/slopes_batch.py`` consumes; the SEC slopes table on
     disk for each well is found by deriving the filename from these
     fields plus the well's date (auto-discovered from the raw CSV).

  2. ``config/ert_panels.yml`` -- which wells you want plotted, and
     the default ERT variant. The variant is independent of campaign
     because ERT data is pre-casing (no campaign concept).

  3. ``data/metadata/ert_wells.csv`` -- geometric mapping of wells to
     (transect, x) points. One row per (well, transect, x); a well
     with three nearby transects has three rows here, which produces
     three figures.

  4. ``config/pipeline.yml`` -- ERT detection and mixing-zone
     parameters (ert.breakpoints.*, ert.bot_mz_rho_threshold).

For every (well, transect, x) association of a well listed in
ert_panels.yml, two figures are produced:

  results/figures/convergence/sec_ert/<campaign>/
      <well>_<transect>_x<x>_linear.png
      <well>_<transect>_x<x>_log10.png

Behaviour: PERMISSIVE by default. Missing slopes-CSV, missing ERT
file, non-converged fits are reported and skipped. Pass ``--strict``
to abort on the first error.

Usage
-----
    uv run python scripts/sec_vs_ert_panels_batch.py --campaign 2022_02

Optional flags
--------------
    --strict
        Abort on first failure (default: skip and continue).
    --slopes-jobs PATH
        Override slopes jobs YAML.
    --ert-panels PATH
        Override ert_panels.yml.
    --ert-wells PATH
        Override ert_wells.csv.
    --pipeline-config PATH
        Override pipeline.yml.
    --raw-dir, --processed-dir, --slopes-dir, --ert-data-dir, --fig-dir
        Override the canonical paths.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from karst_analysis.config import load_config
from karst_analysis.ert.breakpoints import (
    detect_breakpoints_with_seed_discovery,
)
from karst_analysis.ert.io import ErtWellMap, load_ert_1d_traces
from karst_analysis.ert.mixing_zone import select_ert_mixing_zone
from karst_analysis.ert.viz import SecVsErtInputs, plot_sec_vs_ert


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  YAML loaders
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SecJob:
    well: str
    method: str
    trial: str   # e.g. "trial_3"
    n: int

    @property
    def trial_idx(self) -> int:
        m = re.match(r"trial_(\d+)$", self.trial)
        if m is None:
            raise ValueError(f"Unrecognised trial label: {self.trial!r}")
        return int(m.group(1))


def load_slopes_jobs(path: Path) -> tuple[str, dict[str, SecJob]]:
    """Return (campaign, dict mapping well_id -> SecJob)."""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    campaign = str(cfg["campaign"])
    out: dict[str, SecJob] = {}
    for raw in cfg.get("jobs", []):
        job = SecJob(
            well=str(raw["well"]),
            method=str(raw["method"]),
            trial=str(raw["trial"]),
            n=int(raw["n"]),
        )
        out[job.well] = job
    return campaign, out


def load_ert_panels(path: Path) -> tuple[str, list[str]]:
    """Return (default_variant, list_of_wells) from ert_panels.yml.

    Empty/missing wells list means "all wells in ert_wells.csv".
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    variant = str(cfg.get("default_variant", "viz_sharp"))
    wells = list(cfg.get("wells") or [])
    return variant, [str(w) for w in wells]


# ════════════════════════════════════════════════════════════════════
#  File discovery
# ════════════════════════════════════════════════════════════════════
_DATE_RE = re.compile(r"_(\d{8})\.csv$")


def find_well_date(raw_dir: Path, well_id: str) -> Optional[str]:
    """Return YYYYMMDD found in the raw YSI CSV name for the given well.

    Raw filenames look like ``LRS70_D_YSI_20220131.csv`` /
    ``LRS70D_YSI_20220131.csv`` etc.; we just find the trailing
    8-digit run before .csv. None if nothing matches.
    """
    candidates = list(raw_dir.glob(f"*{well_id}*"))
    # Some well_ids look like "LRS70D" but the file name is
    # "LRS70_D_..." -- try a relaxed search.
    if not candidates and len(well_id) >= 4:
        head = well_id[:-1]
        tail = well_id[-1]
        candidates = list(raw_dir.glob(f"*{head}_{tail}*"))
    for p in candidates:
        m = _DATE_RE.search(p.name)
        if m:
            return m.group(1)
    return None


def find_smoothed_csv(
    processed_dir: Path, well_id: str, date_str: str, method: str,
) -> Optional[Path]:
    """Look up the smoothed-cast CSV produced by preprocess_batch.

    Convention (from ``scripts/preprocess_batch.py``): smoothed CSVs
    live under ``<processed_dir>/<method>/`` with filename
    ``<well>_<date>__<method>-<params>.csv``.
    """
    method_subdir = processed_dir / method
    pattern = f"{well_id}_{date_str}__{method}-*.csv"
    matches = list(method_subdir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        log.warning(
            "Multiple smoothed CSVs match %s in %s; using %s",
            pattern, method_subdir, matches[0].name,
        )
        return matches[0]
    # Fallback: try the parent dir without /method/ subdir, in case
    # the user is using a flat layout.
    flat_matches = list(processed_dir.glob(pattern))
    if len(flat_matches) == 1:
        return flat_matches[0]
    if len(flat_matches) > 1:
        log.warning(
            "Multiple smoothed CSVs match %s in flat layout %s; using %s",
            pattern, processed_dir, flat_matches[0].name,
        )
        return flat_matches[0]
    return None


def find_raw_csv(raw_dir: Path, well_id: str, date_str: str) -> Optional[Path]:
    """Look up the raw YSI CSV for a (well, date)."""
    candidates = list(raw_dir.glob(f"*{well_id}*{date_str}*.csv"))
    if not candidates and len(well_id) >= 4:
        head, tail = well_id[:-1], well_id[-1]
        candidates = list(raw_dir.glob(f"*{head}_{tail}*{date_str}*.csv"))
    return candidates[0] if candidates else None


def slopes_csv_path(
    slopes_dir: Path, well_id: str, date_str: str,
    method: str, n: int, trial_idx: int,
) -> Path:
    return slopes_dir / (
        f"{well_id}_{date_str}__slopes-{method}-N{n}-t{trial_idx}.csv"
    )


# ════════════════════════════════════════════════════════════════════
#  Per-job processing
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class JobOutcome:
    well: str
    transect: str
    x: float
    ok: bool
    reason: str = ""


def process_one(
    *,
    well_id: str,
    transect: str,
    x: float,
    variant: str,
    sec_job: SecJob,
    campaign: str,
    raw_dir: Path,
    processed_dir: Path,
    slopes_dir: Path,
    ert_data_dir: Path,
    fig_dir: Path,
    ert_n: int,
    ert_tolerance: float,
    ert_min_distance: float,
    ert_start_seed: int,
    ert_max_seed_attempts: int,
    bot_mz_rho_threshold: float,
    strict: bool,
) -> JobOutcome:
    """Render the two figures (linear, log10) for one association."""
    label = f"{well_id} / {transect}@x{x:g}"
    log.info("processing %s (variant=%s)", label, variant)

    # ── 1) date for this well ──
    date_str = find_well_date(raw_dir, well_id)
    if date_str is None:
        msg = f"raw CSV not found in {raw_dir} for well {well_id}"
        if strict:
            raise FileNotFoundError(msg)
        return JobOutcome(well_id, transect, x, False, msg)

    # ── 2) raw + smoothed SEC ──
    raw_path = find_raw_csv(raw_dir, well_id, date_str)
    if raw_path is None:
        msg = f"raw CSV missing for {well_id} {date_str}"
        if strict:
            raise FileNotFoundError(msg)
        return JobOutcome(well_id, transect, x, False, msg)

    smoothed_path = find_smoothed_csv(
        processed_dir, well_id, date_str, sec_job.method,
    )
    if smoothed_path is None:
        msg = (
            f"smoothed CSV missing for {well_id} {date_str} "
            f"(method={sec_job.method})"
        )
        if strict:
            raise FileNotFoundError(msg)
        return JobOutcome(well_id, transect, x, False, msg)

    # ── 3) slopes CSV ──
    slopes_path = slopes_csv_path(
        slopes_dir, well_id, date_str,
        sec_job.method, sec_job.n, sec_job.trial_idx,
    )
    if not slopes_path.is_file():
        msg = f"slopes CSV missing: {slopes_path}"
        if strict:
            raise FileNotFoundError(msg)
        return JobOutcome(well_id, transect, x, False, msg)

    df_raw = pd.read_csv(raw_path)
    df_sm = pd.read_csv(smoothed_path)
    df_sl = pd.read_csv(slopes_path)

    # ── 4) ERT trace ──
    project_root = ert_data_dir.parent.parent.parent
    try:
        traces = load_ert_1d_traces(
            transect, project_root=project_root,
            x_filter=x, variant_filter=[variant],
        )
    except (FileNotFoundError, ValueError) as exc:
        msg = f"ERT load failed: {exc}"
        if strict:
            raise
        return JobOutcome(well_id, transect, x, False, msg)
    if not traces:
        msg = f"no ERT trace at {transect} x={x} variant={variant}"
        if strict:
            raise FileNotFoundError(msg)
        return JobOutcome(well_id, transect, x, False, msg)
    trace = traces[0]

    # ── 5) ERT detection ──
    z = trace.df["depth_bgl_m"].to_numpy()
    y = trace.df["resistlog10"].to_numpy()
    try:
        fit = detect_breakpoints_with_seed_discovery(
            z, y,
            n_breakpoints=ert_n,
            tolerance=ert_tolerance,
            min_distance=ert_min_distance,
            start_seed=ert_start_seed,
            max_seed_attempts=ert_max_seed_attempts,
        )
    except RuntimeError as exc:
        msg = f"ERT did not converge: {exc}"
        if strict:
            raise
        return JobOutcome(well_id, transect, x, False, msg)

    # ── 6) Mixing zone ──
    z_bp = fit.breakpoints["Breakpoint X Position"].to_numpy()
    rho_bp = np.interp(z_bp, z, trace.df["resist_ohm_m"].to_numpy())
    top_idx, bot_idx = select_ert_mixing_zone(
        z_bp, rho_bp, bot_mz_rho_threshold=bot_mz_rho_threshold,
    )

    # ── 7) Build inputs ──
    vadose_m = float(df_sm["depth_bgl_m"].iloc[0] - df_sm["depth_m"].iloc[0])
    sec_date_str = (
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    )
    inputs = SecVsErtInputs(
        well_id=well_id,
        sec_date_str=sec_date_str,
        z_raw_bgl_m=df_raw.get(
            "Depth from GL (m)", df_raw.get("depth_bgl_m", pd.Series())
        ).to_numpy(),
        sec_raw_uS_cm=df_raw.get(
            "Corrected sp Cond [\u00b5S/cm]",
            df_raw.get("sec_uS_cm", pd.Series()),
        ).to_numpy(),
        z_smooth_bgl_m=df_sm["depth_bgl_m"].to_numpy(),
        sec_smooth_uS_cm=df_sm["sec_uS_cm"].to_numpy(),
        slopes_df=df_sl,
        vadose_m=vadose_m,
        sec_method=sec_job.method,
        sec_n=sec_job.n,
        sec_trial_idx=sec_job.trial_idx,
        ert_trace=trace,
        ert_fit=fit,
        ert_top_mz_idx=top_idx,
        ert_bot_mz_idx=bot_idx,
        ert_bot_mz_threshold=bot_mz_rho_threshold,
    )

    # ── 8) Render and save both scales ──
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scale in ("linear", "log10"):
        fig = plot_sec_vs_ert(inputs, axis_scale=scale)
        out = fig_dir / (
            f"{well_id}_{transect}_x{x:g}_{scale}.png"
        )
        fig.savefig(out, dpi=130, bbox_inches="tight")
        # Free memory.
        import matplotlib.pyplot as plt
        plt.close(fig)
        log.info("  wrote %s", out.name)

    return JobOutcome(well_id, transect, x, True)


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch SEC vs ERT 1D comparison panels."
    )
    p.add_argument("--campaign", required=True)
    p.add_argument("--slopes-jobs", type=Path, default=None)
    p.add_argument("--ert-panels", type=Path,
                   default=Path("config/ert_panels.yml"))
    p.add_argument("--ert-wells", type=Path,
                   default=Path("data/metadata/ert_wells.csv"))
    p.add_argument("--pipeline-config", type=Path, default=None,
                   help="Pipeline YAML (defaults to config/pipeline.yml).")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--slopes-dir", type=Path, default=None)
    p.add_argument("--ert-data-dir", type=Path, default=None,
                   help="Root of ERT raw data (default: data/raw/ert).")
    p.add_argument("--fig-dir", type=Path, default=None)
    p.add_argument("--strict", action="store_true",
                   help="Abort on first error (default: skip and continue).")
    args = p.parse_args(argv)

    campaign = args.campaign

    slopes_jobs_path = args.slopes_jobs or Path(
        f"config/slopes_jobs_{campaign}.yml"
    )
    if not slopes_jobs_path.is_file():
        log.error("slopes_jobs not found: %s", slopes_jobs_path)
        return 1
    slopes_campaign, sec_jobs_by_well = load_slopes_jobs(slopes_jobs_path)
    if slopes_campaign != campaign:
        log.error(
            "campaign mismatch: --campaign=%s but slopes_jobs has %s",
            campaign, slopes_campaign,
        )
        return 1

    if not args.ert_panels.is_file():
        log.error("ert_panels.yml not found: %s", args.ert_panels)
        return 1
    default_variant, wells_filter = load_ert_panels(args.ert_panels)

    if not args.ert_wells.is_file():
        log.error("ert_wells.csv not found: %s", args.ert_wells)
        return 1
    well_map = ErtWellMap.from_csv(args.ert_wells)

    cfg = load_config(args.pipeline_config) if args.pipeline_config \
        else load_config()
    ert_cfg = cfg.get("ert", {})
    bp_cfg = ert_cfg.get("breakpoints", {})
    ert_n = int(bp_cfg.get("max_breakpoints", 15))
    ert_tolerance = float(bp_cfg.get("tolerance", 1.0e-5))
    ert_min_distance = float(bp_cfg.get("min_distance", 0.01))
    ert_start_seed = int(bp_cfg.get("start_seed", 0))
    ert_max_seed_attempts = int(bp_cfg.get("max_seed_attempts", 20))
    bot_mz_rho_threshold = float(ert_cfg.get("bot_mz_rho_threshold", 25.0))

    raw_dir = args.raw_dir or Path(f"data/raw/sec/{campaign}")
    processed_dir = args.processed_dir or Path(
        f"data/processed/sec/{campaign}"
    )
    slopes_dir = args.slopes_dir or Path(f"data/slopes/{campaign}")
    ert_data_dir = args.ert_data_dir or Path("data/raw/ert")
    fig_dir = args.fig_dir or Path(
        f"results/figures/convergence/sec_ert/{campaign}"
    )

    # Resolve target wells: either explicit list from ert_panels.yml,
    # or all wells present in ert_wells.csv.
    if wells_filter:
        target_wells = wells_filter
    else:
        target_wells = sorted({a.well_id for a in well_map.associations})

    log.info("campaign=%s, %d target wells, fig_dir=%s",
             campaign, len(target_wells), fig_dir)

    outcomes: list[JobOutcome] = []
    for well_id in target_wells:
        sec_job = sec_jobs_by_well.get(well_id)
        if sec_job is None:
            log.warning(
                "no SEC slopes_job for well %s in %s; skipping",
                well_id, slopes_jobs_path.name,
            )
            continue
        assocs = well_map.transects_for_well(well_id)
        if not assocs:
            log.warning(
                "no ert_wells.csv entry for well %s; skipping",
                well_id,
            )
            continue
        for a in assocs:
            outcome = process_one(
                well_id=well_id,
                transect=a.transect,
                x=a.x,
                variant=default_variant,
                sec_job=sec_job,
                campaign=campaign,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                slopes_dir=slopes_dir,
                ert_data_dir=ert_data_dir,
                fig_dir=fig_dir,
                ert_n=ert_n,
                ert_tolerance=ert_tolerance,
                ert_min_distance=ert_min_distance,
                ert_start_seed=ert_start_seed,
                ert_max_seed_attempts=ert_max_seed_attempts,
                bot_mz_rho_threshold=bot_mz_rho_threshold,
                strict=args.strict,
            )
            outcomes.append(outcome)

    n_ok = sum(1 for o in outcomes if o.ok)
    n_skip = sum(1 for o in outcomes if not o.ok)
    log.info("DONE: %d ok, %d skipped", n_ok, n_skip)
    if n_skip:
        log.info("skipped:")
        for o in outcomes:
            if not o.ok:
                log.info("  %s / %s @ x=%g  -- %s",
                         o.well, o.transect, o.x, o.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
