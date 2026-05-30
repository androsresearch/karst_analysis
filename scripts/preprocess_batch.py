"""Batch preprocessing CLI for SEC profiles — config-driven.

Reads parameters from a YAML config file (see ``config/pipeline.yml``).
Missing keys fall back to ``config/pipeline_default.yml``.

Usage
-----
Default config (``config/pipeline.yml``):

    uv run python scripts/preprocess_batch.py \\
        --input data/raw/sec/2022_02 \\
        --output data/processed/sec/2022_02

Custom config (e.g. for sensitivity analysis):

    uv run python scripts/preprocess_batch.py \\
        --input data/raw/sec/2022_02 \\
        --output data/processed/sec/2022_02_window21 \\
        --config config/sensitivity_window21.yml

The script writes one processed CSV per (well, method) pair, plus
diagnostic PNGs in ``results/figures/diagnostic/<campaign>/``. Every
run is logged in ``results/runs.csv`` with the full parameter set
recorded in ``params_json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from karst_analysis.config import (
    ConfigError, default_config_path, load_config, params_for_run_ledger,
)
from karst_analysis.corrections import get_vadose_thickness, load_well_metadata
from karst_analysis.io import parse_well_filename
from karst_analysis.runs import register_run
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.preprocessing import process_lowess, process_savgol
from karst_analysis.sec.viz import plot_diagnostic


# ─────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────
class _ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        c = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{c}{record.levelname}{self.RESET}"
        record.msg = f"{c}{record.msg}{self.RESET}"
        return super().format(record)


def _setup_logger(name: str = "preprocess") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(_ColoredFormatter("[%(asctime)s] %(levelname)s: %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


# ─────────────────────────────────────────────────────────────────────────
#  Per-file processing
# ─────────────────────────────────────────────────────────────────────────
def _process_one(
    csv_path: Path,
    output_root: Path,
    cfg: dict,
    *,
    metadata: Optional[pd.DataFrame],
    logger: logging.Logger,
) -> list[dict]:
    """Process a single CSV with each method requested in ``cfg``.

    Reads all parameters (cleaning, savgol, lowess, log10, etc.) from
    the config dict — no hardcoded values.
    """
    pp = cfg["preprocessing"]
    methods: list[str] = list(pp["methods"])
    campaign: str = cfg["campaign"]

    try:
        info = parse_well_filename(csv_path)
    except ValueError as e:
        logger.error(f"Skipping {csv_path.name}: {e}")
        return [{"file": csv_path.name, "status": "failed", "error": str(e)}]

    df_raw = load_ysi_csv(csv_path)

    vadose: Optional[float] = None
    if metadata is not None and info.well_id in metadata.index:
        vadose = float(metadata.loc[info.well_id, "vadose_thickness_m"])
        logger.info(f"Vadose thickness for {info.well_id}: {vadose} m")
    else:
        logger.warning(
            f"No vadose thickness for {info.well_id}; depth_bgl_m omitted."
        )

    # Common kwargs (cleaning + log10 + vadose) — independent of method
    common_kwargs = dict(
        apply_depth_adjustment=pp["apply_depth_adjustment"],
        depth_adjustment=pp["depth_adjustment"],
        depth_adjustment_method=pp["depth_adjustment_method"],
        apply_monotonic_descent_filter=pp["apply_monotonic_descent_filter"],
        monotonic_descent_tolerance=pp["monotonic_descent_tolerance"],
        dz=pp["dz"],
        dz_method=pp["dz_method"],
        apply_log10=pp["apply_log10"],
        vadose_thickness_m=vadose,
        logger=logger,
    )

    results = []
    for m in methods:
        out_dir = output_root / m
        from karst_analysis.io import resolve_figure_dir
        fig_dir = resolve_figure_dir("diagnostic", campaigns=[campaign])

        params = params_for_run_ledger(cfg, "preprocessing", method=m)

        if m == "savgol":
            sp = pp["savgol"]
            run_fn = lambda df: process_savgol(
                df,
                **common_kwargs,
                savgol_window=sp["window"],
                savgol_order=sp["order"],
                savgol_segmented=sp["segmented"],
                savgol_gradient_factor=sp["gradient_factor"],
                savgol_min_gradient_threshold=sp["min_gradient_threshold"],
            )
        elif m == "lowess":
            lp = pp["lowess"]
            run_fn = lambda df: process_lowess(
                df,
                **common_kwargs,
                lowess_frac=lp["frac"],
                lowess_degree=lp["degree"],
                lowess_iter=lp["n_robust_iter"],
                apply_pava=lp["apply_pava"],
            )
        else:
            logger.error(f"Unknown method '{m}' in config; skipping.")
            continue

        try:
            with register_run(
                stage="smoothing",
                well_id=info.well_id,
                date=info.date,
                input_file=str(csv_path),
                params=params,
                output_dir=out_dir,
            ) as run:
                df_out, stats = run_fn(df_raw)
                df_out.to_csv(run.output_path, index=False)

                # Diagnostic figure
                depth_col = "depth_m" if "depth_m" in df_raw.columns else "Vertical Position [m]"
                sec_col = "sec_uS_cm" if "sec_uS_cm" in df_raw.columns else "Corrected sp Cond [µS/cm]"
                fig_dir.mkdir(parents=True, exist_ok=True)
                fig_path = fig_dir / f"{info.well_id}_{info.date}__{run.method_signature}.png"

                # Vadose-zone thickness so figures render in BGL
                # (canonical datum). Falls back to water-table datum
                # with honest label if the well is missing from
                # wells.csv.
                try:
                    vadose_offset_m = get_vadose_thickness(info.well_id)
                except (KeyError, FileNotFoundError):
                    vadose_offset_m = 0.0

                plot_diagnostic(
                    df_raw[depth_col].to_numpy(), df_raw[sec_col].to_numpy(),
                    df_out[depth_col].to_numpy(), df_out[sec_col].to_numpy(),
                    output_path=fig_path,
                    title=f"{info.well_id} {info.date} — {m}",
                    method_info=run.method_signature,
                    vadose_offset_m=vadose_offset_m,
                )

                run.note = ""
                logger.info(f"✓ {info.well_id} {m}: {run.output_path.name}")
                results.append({
                    "file": csv_path.name, "method": m,
                    "status": "success", "run_id": run.run_id,
                    "output": str(run.output_path),
                })
        except Exception as e:
            logger.error(f"✗ {info.well_id} {m}: {e}")
            results.append({
                "file": csv_path.name, "method": m,
                "status": "failed", "error": str(e),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing raw CSVs.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output root directory (per-method subdirs are created under it).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to user YAML config. Defaults to config/pipeline.yml. "
             "Falls back to config/pipeline_default.yml for missing keys.",
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Skip vadose-zone correction (no depth_bgl_m).",
    )
    args = parser.parse_args()

    # Resolve config path: explicit --config wins, else pipeline.yml, else default-only.
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

    logger = _setup_logger()
    if cfg_path is not None:
        logger.info(f"Config: {cfg_path}")
    else:
        logger.info(f"Config: defaults only ({default_config_path()})")

    in_folder = Path(args.input)
    if not in_folder.is_dir():
        print(f"ERROR: input folder not found: {in_folder}", file=sys.stderr)
        return 1
    out_root = Path(args.output)
    csvs = sorted(in_folder.glob("*.csv"))
    if not csvs:
        print(f"ERROR: no CSV files in {in_folder}", file=sys.stderr)
        return 1

    metadata = None
    if not args.no_metadata:
        try:
            metadata = load_well_metadata()
            logger.info(f"Loaded metadata for {len(metadata)} wells.")
        except FileNotFoundError as e:
            logger.warning(str(e))

    methods = cfg["preprocessing"]["methods"]
    logger.info(
        f"Processing {len(csvs)} files with methods={methods} "
        f"(campaign={cfg['campaign']})."
    )
    start = datetime.now()
    all_results = []

    for csv_path in tqdm(csvs, desc="Files", unit="file"):
        results = _process_one(
            csv_path, out_root, cfg,
            metadata=metadata, logger=logger,
        )
        all_results.extend(results)

    elapsed = (datetime.now() - start).total_seconds()
    success = sum(1 for r in all_results if r["status"] == "success")
    fail = sum(1 for r in all_results if r["status"] == "failed")

    print("\n" + "=" * 60)
    print(f" SUMMARY — {success} succeeded / {fail} failed in {elapsed:.1f} s")
    print("=" * 60)
    for r in all_results:
        if r["status"] == "failed":
            print(f"  ✗ {r['file']} ({r.get('method', '?')}): {r.get('error')}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
