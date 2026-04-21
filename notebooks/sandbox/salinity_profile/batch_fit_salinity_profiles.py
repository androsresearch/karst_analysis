"""Batch-fit the Huang et al. (2024) salinity sigmoid over a folder of CSV
profiles.

Run directly with no CLI arguments — paths and options are configured in
the CONFIGURATION block below:

    uv run python notebooks/sandbox/salinity_profile/batch_fit_salinity_profiles.py

For each CSV in ``INPUT_FOLDER`` the script
  1. loads the profile (auto-detects µS/cm vs mS/cm units),
  2. applies the preprocessing chain (trim / pre-resample / LOWESS / PAVA /
     final resample),
  3. fits the modified van Genuchten sigmoid,
  4. computes mixing-zone attributes,
  5. writes {stem}_fit_plot.png and {stem}_fit_params.json to OUTPUT_FOLDER.

A single ``fitting_summary.csv`` and ``batch_run.log`` are written at the end.
Failures in one file are logged and do not abort the batch.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Force UTF-8 so the µ and → characters printed by fit_salinity_profile
# don't crash on Windows consoles (cp1252 default).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_salinity_profile as fsp  # noqa: E402


# CONFIGURATION  --  edit, save, run
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Input / Output paths (uncomment the preset you want)
# Preset A -- priority wells (AW5, AW6, BW3, LRS69_R, LRS70_R). 
INPUT_FOLDER  = _REPO_ROOT / "data" / "raw_priority"
OUTPUT_FOLDER = _REPO_ROOT / "results" / "salinity_fits" / "priority"

# Preset B -- LRS69 
# INPUT_FOLDER  = _REPO_ROOT / "data" / "raw_lrs69"
# OUTPUT_FOLDER = _REPO_ROOT / "results" / "salinity_fits" / "lrs69"

# Preset C -- LRS70.
# INPUT_FOLDER  = _REPO_ROOT / "data" / "raw_lrs70"
# OUTPUT_FOLDER = _REPO_ROOT / "results" / "salinity_fits" / "lrs70"

PATTERN = "*.csv"

# Fitting bounds mode
#   "open"    -- original non-negativity bounds (backward compatible).
#                May yield numerically-finite but physically absurd
#                attributes (W ~ 1e40 m) for sharp transitions because
#                `m` can drift toward 0 (step-function regime).
#   "bounded" -- physically motivated upper bounds (RECOMMENDED).
BOUNDS_MODE = "bounded"

# Preprocessing (LRS70 notebook defaults)
# Set any of these to None to disable that step.
TRIM_ABOVE          = -1.0    # [m]       elevation threshold
PRERESAMPLE_Z       = 0.01    # [m]       grid step before smoothing
SMOOTH_LOWESS_FRAC  = 0.03    # fraction  LOWESS window size
ENFORCE_MONOTONIC   = True    # apply PAVA after smoothing

# Post-smoothing resampling (only one takes effect; EC wins over Z).
RESAMPLE_Z          = None    # [m]
RESAMPLE_EC         = None    # [mS/cm]

# Output options 
SAVE_PLOT           = True
SAVE_PROCESSED_CSV  = False

_SUMMARY_COLUMNS = [
    "file", "status", "bounds_mode", "n_points_raw", "n_points_used",
    "alpha", "n", "m", "Cf", "Cs", "R2", "RMSE",
    "z_mid", "s_mid", "W", "r_f", "r_s", "z_5mS",
    "EC_min_fit", "EC_max_fit", "converged", "nfev", "error_message",
]


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("batch_fit_salinity_profiles")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _preprocess(z: np.ndarray, EC: np.ndarray,
                logger: logging.Logger
                ) -> tuple[np.ndarray, np.ndarray, int, int]:
    n_raw = len(z)
    steps: list[str] = []

    if TRIM_ABOVE is not None:
        n0 = len(z)
        z, EC = fsp.trim_shallow(z, EC, min_elev_m=TRIM_ABOVE)
        steps.append(f"trim<{TRIM_ABOVE}m {n0}->{len(z)}")

    if PRERESAMPLE_Z is not None:
        n0 = len(z)
        z, EC = fsp.resample_uniform_z(z, EC, step_m=PRERESAMPLE_Z)
        steps.append(f"preresample({PRERESAMPLE_Z}m) {n0}->{len(z)}")

    if SMOOTH_LOWESS_FRAC is not None:
        EC = fsp.lowess_smooth(z, EC, frac=SMOOTH_LOWESS_FRAC)
        steps.append(f"lowess(frac={SMOOTH_LOWESS_FRAC})")

    if ENFORCE_MONOTONIC:
        before = EC.copy()
        EC = fsp.enforce_monotonic_with_depth(z, EC)
        n_changed = int(np.sum(np.abs(EC - before) > 1e-9))
        steps.append(f"pava(adj={n_changed}/{len(EC)})")

    if RESAMPLE_EC is not None:
        n0 = len(z)
        z, EC = fsp.resample_uniform_EC(z, EC, step_mS=RESAMPLE_EC)
        steps.append(f"resample_EC({RESAMPLE_EC}) {n0}->{len(z)}")
    elif RESAMPLE_Z is not None:
        n0 = len(z)
        z, EC = fsp.resample_uniform_z(z, EC, step_m=RESAMPLE_Z)
        steps.append(f"resample_z({RESAMPLE_Z}) {n0}->{len(z)}")

    logger.info("  preprocess: " + " | ".join(steps))
    return z, EC, n_raw, len(z)


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _empty_row(csv_path: Path) -> dict:
    row = {col: None for col in _SUMMARY_COLUMNS}
    row["file"] = csv_path.name
    row["status"] = "failed"
    row["bounds_mode"] = BOUNDS_MODE
    return row


def _process_one(csv_path: Path, output_dir: Path,
                 logger: logging.Logger) -> dict:
    result = _empty_row(csv_path)
    stem = csv_path.stem
    try:
        z_raw, EC_raw = fsp.load_profile(csv_path)
        z, EC, n_raw, n_used = _preprocess(z_raw, EC_raw, logger)
        result["n_points_raw"] = n_raw
        result["n_points_used"] = n_used

        if n_used < 10:
            raise RuntimeError(f"only {n_used} points after preprocessing")

        fit = fsp.fit_profile(z, EC, bounds=BOUNDS_MODE)
        attrs = fsp.mixing_zone_attributes(
            fit, z_range=(float(z.min()), float(z.max())))

        json_path = output_dir / f"{stem}_fit_params.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                _json_safe({"bounds_mode": BOUNDS_MODE,
                            "fit": fit, "attrs": attrs}),
                fh, indent=2, ensure_ascii=False)

        if SAVE_PLOT:
            png_path = output_dir / f"{stem}_fit_plot.png"
            fsp.plot_fit(z, EC, fit, attrs=attrs,
                         raw_overlay=(z_raw, EC_raw),
                         data_label="Observed (post-processed)",
                         output_path=png_path, show=False)

        if SAVE_PROCESSED_CSV:
            proc_path = output_dir / f"{stem}_processed.csv"
            pd.DataFrame({"elevation_m": z,
                          "EC_mS_per_cm": EC}
                         ).to_csv(proc_path, index=False)

        result.update({
            "status": "success",
            "alpha": fit["alpha"], "n": fit["n"], "m": fit["m"],
            "Cf": fit["Cf"], "Cs": fit["Cs"],
            "R2": fit["R2"], "RMSE": fit["RMSE"],
            "z_mid": attrs["z_mid [m]"],
            "s_mid": attrs["s_mid [mS/cm / m]"],
            "W": attrs["Mixing zone thickness W [m]"],
            "r_f": attrs["r_f (fresh->brackish)"],
            "r_s": attrs["r_s (brackish->salt)"],
            "z_5mS": attrs["z_5mS/cm isochlor [m]"],
            "EC_min_fit": fit["EC_min_fit"],
            "EC_max_fit": fit["EC_max_fit"],
            "converged": fit["converged"],
            "nfev": fit["nfev"],
        })
        logger.info(
            f"  OK   R2={fit['R2']:.4f} RMSE={fit['RMSE']:.3f} mS/cm "
            f"z_mid={result['z_mid']:.2f} W={result['W']:.2f}"
        )
    except Exception as e:
        result["error_message"] = str(e)
        logger.error(f"  FAIL {csv_path.name}: {e}")
        logger.debug(traceback.format_exc())
    return result


def _write_summary(results: list[dict], output_dir: Path) -> Path:
    df = pd.DataFrame(results, columns=_SUMMARY_COLUMNS)
    out_path = output_dir / "fitting_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _print_summary(results: list[dict], logger: logging.Logger) -> None:
    ok = [r for r in results if r["status"] == "success"]
    fail = [r for r in results if r["status"] == "failed"]
    logger.info(f"SUMMARY: {len(ok)} ok | {len(fail)} failed | "
                f"{len(results)} total")
    for r in fail:
        logger.info(f"  FAIL {r['file']}: {r['error_message']}")


def main() -> None:
    if not INPUT_FOLDER.is_dir():
        sys.exit(f"ERROR: input folder not found: {INPUT_FOLDER}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(OUTPUT_FOLDER / "batch_run.log")

    csv_files = sorted(INPUT_FOLDER.glob(PATTERN))
    if not csv_files:
        sys.exit(f"ERROR: no files matching {PATTERN} in {INPUT_FOLDER}")

    logger.info(f"Input : {INPUT_FOLDER}  "
                f"({len(csv_files)} files matching {PATTERN})")
    logger.info(f"Output: {OUTPUT_FOLDER}")
    logger.info(f"Bounds mode: {BOUNDS_MODE}")
    logger.info(
        "Preprocess: "
        f"trim={'off' if TRIM_ABOVE is None else TRIM_ABOVE} | "
        f"preresample={'off' if PRERESAMPLE_Z is None else PRERESAMPLE_Z} | "
        f"lowess={'off' if SMOOTH_LOWESS_FRAC is None else SMOOTH_LOWESS_FRAC} | "
        f"monotonic={ENFORCE_MONOTONIC} | "
        f"resample_ec={RESAMPLE_EC} | resample_z={RESAMPLE_Z}"
    )

    start = datetime.now()
    results: list[dict] = []
    for csv in tqdm(csv_files, desc="Fitting", unit="file"):
        logger.info(f"Processing {csv.name}")
        results.append(_process_one(csv, OUTPUT_FOLDER, logger))

    summary_path = _write_summary(results, OUTPUT_FOLDER)
    logger.info(f"Wrote {summary_path}")
    _print_summary(results, logger)
    logger.info(f"Elapsed: {(datetime.now() - start).total_seconds():.1f} s")


if __name__ == "__main__":
    main()
