"""Overnight batch: BIC sweep for every (well, smoothing method) pair.

For each raw CSV under ``--raw-dir``, this script:
  1. Smooths it with **both** SavGol and LOWESS pipelines (using the
     vadose-zone thickness from ``data/metadata/wells.csv``).
  2. Saves the two smoothed CSVs under ``data/processed/sec/<campaign>/``.
  3. Runs ``best_n_breakpoints`` with ``max_breakpoints=N`` and
     ``n_trials=T`` on the log10 conductivity of each smoothed profile.
  4. Saves one JSON per (well, method) pair in
     ``data/breakpoints/<campaign>/{well_id}_{date}__bp-{method}-max{N}-t{T}.json``.
  5. Records each run in ``results/runs.csv``.
  6. (optional) Generates per-N comparison figures (savgol vs lowess) for
     each well in ``results/figures/breakpoints/<campaign>/<well_id>_compare_N1to{N}/``.

Outputs are append-only: re-running with the same parameters overwrites
the JSON for that pair (same filename) but adds a fresh row in runs.csv.

Typical overnight invocation
----------------------------
    uv run python scripts/breakpoints_batch.py \\
        --raw-dir   data/raw/sec/2022_02 \\
        --campaign  2022_02 \\
        --max-bp    10 \\
        --n-trials  3 \\
        --plot

Quick test (single well, 1 trial, no plots)
-------------------------------------------
    uv run python scripts/breakpoints_batch.py \\
        --raw-dir   data/raw/sec/2022_02 \\
        --campaign  2022_02 \\
        --only      LRS70D \\
        --max-bp    10 \\
        --n-trials  1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from karst_analysis.corrections import load_well_metadata
from karst_analysis.io import parse_well_filename
from karst_analysis.runs import register_run
from karst_analysis.sec.breakpoints import best_n_breakpoints
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.preprocessing import process_lowess, process_savgol
from karst_analysis.sec.viz import (
    load_bic_json,
    plot_breakpoints_compare_methods,
)


# ─────────────────────────────────────────────────────────────────────────
#  JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────
def _serialise_results(results: dict) -> dict:
    """Convert the nested results dict to JSON-safe form (handles tuples)."""
    out = {}
    for tname, info_t in results.items():
        df = info_t["df"].copy()
        if "estimates" in df.columns:
            def _clean(est):
                if est is None:
                    return None
                if isinstance(est, dict):
                    cleaned = {}
                    for k, v in est.items():
                        if isinstance(v, dict):
                            d = {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                 for kk, vv in v.items()}
                            cleaned[k] = d
                        else:
                            cleaned[k] = v
                    return cleaned
                return est
            df["estimates"] = df["estimates"].apply(_clean)
        out[tname] = {
            "df": df.to_dict(),
            "best_n_breakpoint_bic": int(info_t["best_n_breakpoint_bic"]),
            "min_bic_n_breakpoint":  int(info_t["min_bic_n_breakpoint"]),
            "best_n_breakpoint_rss": int(info_t["best_n_breakpoint_rss"]),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Per-well processing
# ─────────────────────────────────────────────────────────────────────────
def process_one_well(
    csv_path: Path,
    *,
    proc_root: Path,
    bp_dir: Path,
    fig_dir: Path,
    metadata: pd.DataFrame,
    max_bp: int,
    n_trials: int,
    tolerance: float,
    min_distance: float,
    use_log10: bool,
    plot: bool,
    logger: logging.Logger,
) -> dict:
    """Run smoothing + BIC sweep for a single raw CSV.

    Returns
    -------
    dict
        Summary with timings and output paths.
    """
    info = parse_well_filename(csv_path)
    df_raw = load_ysi_csv(csv_path, standardise=True)

    if info.well_id in metadata.index:
        vadose = float(metadata.loc[info.well_id, "vadose_thickness_m"])
    else:
        logger.warning(f"{info.well_id}: no vadose value in metadata; depth_bgl_m omitted.")
        vadose = None

    # ── 1) smooth with both methods ─────────────────────────────────
    df_savgol, _ = process_savgol(df_raw, vadose_thickness_m=vadose)
    df_lowess, _ = process_lowess(df_raw, vadose_thickness_m=vadose)

    proc_root.mkdir(parents=True, exist_ok=True)
    (proc_root / "savgol").mkdir(exist_ok=True)
    (proc_root / "lowess").mkdir(exist_ok=True)
    sav_path = proc_root / "savgol" / f"{info.well_id}_{info.date}__savgol-w11-o3-seg.csv"
    low_path = proc_root / "lowess" / f"{info.well_id}_{info.date}__lowess-f0.05-i2-pava.csv"
    df_savgol.to_csv(sav_path, index=False)
    df_lowess.to_csv(low_path, index=False)

    y_col = "log10_sec_uS_cm" if use_log10 else "sec_uS_cm"

    # ── 2) BIC sweep for each method ────────────────────────────────
    json_paths = {}
    timings = {}
    for method, df_smooth in [("savgol", df_savgol), ("lowess", df_lowess)]:
        mask = df_smooth[y_col].notna()
        x = df_smooth.loc[mask, "depth_m"].to_numpy()
        y = df_smooth.loc[mask, y_col].to_numpy()

        params = {
            "method": "breakpoints",
            "smoothing_method": method,
            "max_breakpoints": max_bp,
            "n_trials": n_trials,
            "tolerance": tolerance,
            "min_distance": min_distance,
            "y_space": "log10" if use_log10 else "linear",
        }

        t0 = time.time()
        try:
            with register_run(
                stage="breakpoints",
                well_id=info.well_id, date=info.date,
                input_file=str(csv_path), params=params,
                output_dir=bp_dir, extension="json",
            ) as run:
                results = best_n_breakpoints(
                    x, y,
                    max_breakpoints=max_bp,
                    n_trials=n_trials,
                    tolerance=tolerance,
                    min_distance=min_distance,
                )
                with open(run.output_path, "w") as f:
                    json.dump(_serialise_results(results), f, default=str, indent=2)
                run.note = f"batch run {datetime.now().isoformat(timespec='seconds')}"
                json_paths[method] = run.output_path
        except Exception as e:
            logger.error(f"{info.well_id} {method}: {type(e).__name__}: {e}")
            json_paths[method] = None
        timings[method] = time.time() - t0
        if json_paths[method]:
            logger.info(
                f"  ✓ {info.well_id} {method:<6} fit {timings[method]:>6.1f} s "
                f"→ {json_paths[method].name}"
            )

    # ── 3) per-N comparison figures (optional) ──────────────────────
    fig_paths = []
    if plot and json_paths.get("savgol") and json_paths.get("lowess"):
        well_fig_dir = fig_dir / f"{info.well_id}_compare_N1to{max_bp}"
        well_fig_dir.mkdir(parents=True, exist_ok=True)

        bic_savgol = load_bic_json(json_paths["savgol"])
        bic_lowess = load_bic_json(json_paths["lowess"])

        z_raw = df_raw["depth_m"].to_numpy()
        EC_raw = df_raw["sec_uS_cm"].to_numpy()
        z_savgol = df_savgol["depth_m"].to_numpy()
        EC_savgol = df_savgol["sec_uS_cm"].to_numpy()
        z_lowess = df_lowess["depth_m"].to_numpy()
        EC_lowess = df_lowess["sec_uS_cm"].to_numpy()

        for n in range(1, max_bp + 1):
            try:
                fp = well_fig_dir / f"{info.well_id}_{info.date}_N{n:02d}.png"
                plot_breakpoints_compare_methods(
                    z_raw=z_raw, EC_raw=EC_raw,
                    z_left=z_savgol, EC_left=EC_savgol,
                    bic_data_left=bic_savgol, label_left="savgol",
                    z_right=z_lowess, EC_right=EC_lowess,
                    bic_data_right=bic_lowess, label_right="lowess",
                    n_breakpoints=n,
                    trial="trial_1",
                    output_path=fp,
                    title=f"{info.well_id} {info.date}",
                )
                fig_paths.append(fp)
            except Exception as e:
                logger.warning(f"  N={n} plot failed: {e}")

        logger.info(f"  → {len(fig_paths)} comparison figures in {well_fig_dir}")

    return {
        "well_id": info.well_id,
        "date": info.date,
        "json_savgol": json_paths.get("savgol"),
        "json_lowess": json_paths.get("lowess"),
        "fit_time_savgol_s": timings.get("savgol"),
        "fit_time_lowess_s": timings.get("lowess"),
        "n_figures": len(fig_paths),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("breakpoints_batch")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, help="Folder with raw CSVs.")
    p.add_argument("--campaign", default=None,
                   help="Campaign tag for output paths. If omitted, "
                        "read from the config (default: 2022_02).")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (default: config/pipeline.yml). "
                        "CLI args below, when given explicitly, override the config.")
    # NOTE: defaults below are sentinel None — that lets us tell whether
    # the user explicitly passed a CLI value (overrides config) vs. left
    # it absent (use config value).
    p.add_argument("--max-bp", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--min-distance", type=float, default=None)
    p.add_argument("--no-log10", action="store_true",
                   help="Run breakpoints on linear SEC instead of log10. "
                        "If passed, overrides config.")
    p.add_argument("--only", action="append", default=None,
                   help="Restrict to specific well_id(s) (e.g. --only LRS70D). "
                        "Repeatable.")
    p.add_argument("--plot", action="store_true",
                   help="Also generate per-N savgol-vs-lowess comparison figures.")
    p.add_argument("--proc-root", default=None,
                   help="Override processed-CSV root (default: data/processed/sec/<campaign>).")
    p.add_argument("--bp-dir", default=None,
                   help="Override breakpoints JSON dir (default: data/breakpoints/<campaign>).")
    p.add_argument("--fig-dir", default=None,
                   help="Override figure dir (default: results/figures/breakpoints/<campaign>).")
    args = p.parse_args()

    # ── Load config (defaults + user overrides), then apply CLI overrides ──
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

    bp_cfg = cfg["breakpoints"]
    campaign = args.campaign if args.campaign is not None else cfg["campaign"]
    max_bp = args.max_bp if args.max_bp is not None else bp_cfg["max_breakpoints"]
    n_trials = args.n_trials if args.n_trials is not None else bp_cfg["n_trials"]
    tolerance = args.tolerance if args.tolerance is not None else bp_cfg["tolerance"]
    min_distance = args.min_distance if args.min_distance is not None else bp_cfg["min_distance"]
    # --no-log10 is a flag: if user passed it, override config; else use config.
    use_log10 = (not args.no_log10) if args.no_log10 else bp_cfg["use_log10"]

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: raw-dir not found: {raw_dir}", file=sys.stderr)
        return 1

    proc_root = Path(args.proc_root) if args.proc_root else \
                Path("data/processed/sec") / campaign
    bp_dir    = Path(args.bp_dir)    if args.bp_dir    else \
                Path("data/breakpoints") / campaign
    if args.fig_dir:
        fig_dir = Path(args.fig_dir)
    else:
        from karst_analysis.io import resolve_figure_dir
        fig_dir = resolve_figure_dir("breakpoints", campaigns=[campaign])

    for d in [proc_root, bp_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metadata = load_well_metadata()
    logger = _setup_logger()

    csvs = sorted(raw_dir.glob("*.csv"))
    if args.only:
        wanted = set(args.only)
        filtered = []
        for c in csvs:
            try:
                info = parse_well_filename(c)
                if info.well_id in wanted:
                    filtered.append(c)
            except ValueError:
                continue
        csvs = filtered

    print()
    print("=" * 72)
    print(" BREAKPOINTS BATCH")
    print("=" * 72)
    print(f"  config        : {cfg_path if cfg_path else 'defaults only'}")
    print(f"  campaign      : {campaign}")
    print(f"  raw-dir       : {raw_dir}")
    print(f"  files to run  : {len(csvs)}")
    print(f"  max-bp        : {max_bp}")
    print(f"  n-trials      : {n_trials}")
    print(f"  y-space       : {'log10' if use_log10 else 'linear'}")
    print(f"  plot per N    : {'YES' if args.plot else 'no'}")
    print(f"  bp output dir : {bp_dir}")
    if args.plot:
        print(f"  fig output dir: {fig_dir}")
    print("=" * 72)
    print()

    if not csvs:
        print("Nothing to do.")
        return 0

    start = datetime.now()
    summaries = []
    for i, csv_path in enumerate(csvs, 1):
        logger.info(f"[{i}/{len(csvs)}] {csv_path.name}")
        try:
            summaries.append(process_one_well(
                csv_path,
                proc_root=proc_root, bp_dir=bp_dir, fig_dir=fig_dir,
                metadata=metadata,
                max_bp=max_bp, n_trials=n_trials,
                tolerance=tolerance, min_distance=min_distance,
                use_log10=use_log10,
                plot=args.plot, logger=logger,
            ))
        except Exception as e:
            logger.error(f"  ✗ {csv_path.name}: {type(e).__name__}: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    print()
    print("=" * 72)
    print(f" SUMMARY — {len(summaries)} wells in {elapsed/60:.1f} min")
    print("=" * 72)
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
