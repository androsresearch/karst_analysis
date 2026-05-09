"""Stable accessor functions for SEC artefacts.

Naming convention on disk
-------------------------
Smoothed profiles:
    data/processed/sec/<campaign>/<method>/{well_id}_{date}__{sig}.csv

Breakpoint BIC sweeps:
    data/breakpoints/<campaign>/{well_id}_{date}__bp-{method}-max{N}-t{T}.json

Where ``<method>`` is "savgol" or "lowess", ``<sig>`` is the method
signature, and ``{N}/{T}`` are the sweep parameters.

The functions in this module take a (well_id, campaign, smoothing)
triple, locate the matching artefact on disk, and return a tidy
DataFrame. If multiple files match (e.g. several BIC sweeps with
different ``max_breakpoints``), the most recent by mtime wins by
default; this can be overridden with the ``run_id`` argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.corrections import load_well_metadata
from karst_analysis.sec.viz.breakpoints_overlay import load_bic_json


# Where the project root is, relative to the caller. We do NOT hard-code
# absolute paths — instead, every function takes a ``project_root``
# argument that defaults to the current working directory.
DEFAULT_PROJECT_ROOT = Path.cwd

VALID_SMOOTHINGS = ("savgol", "lowess")


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────
def _project_root(project_root: Optional[Path | str]) -> Path:
    return Path(project_root) if project_root is not None else Path.cwd()


def _validate_smoothing(smoothing: str) -> None:
    if smoothing not in VALID_SMOOTHINGS:
        raise ValueError(
            f"smoothing must be one of {VALID_SMOOTHINGS}; got '{smoothing}'."
        )


def _vadose_for(well_id: str, project_root: Path) -> Optional[float]:
    """Return the vadose-zone thickness for a well, or None if unknown."""
    md_path = project_root / "data" / "metadata" / "wells.csv"
    if not md_path.exists():
        return None
    md = load_well_metadata(md_path)
    if well_id not in md.index:
        return None
    return float(md.loc[well_id, "vadose_thickness_m"])


def _select_best_trial_for_n(bic_data: dict, n: int) -> str:
    """Select the trial whose BIC at N is lowest."""
    best_trial, best_bic = None, np.inf
    for trial in bic_data:
        df = pd.DataFrame(bic_data[trial]["df"])
        match = df[df["n_breakpoints"] == n]
        if not len(match):
            continue
        bic = match.iloc[0].get("bic")
        if bic is not None and float(bic) < best_bic:
            best_bic = float(bic)
            best_trial = trial
    if best_trial is None:
        raise ValueError(f"No trial contains N={n}.")
    return best_trial


def _resolve_processed_csv(
    well_id: str,
    campaign: str,
    smoothing: str,
    project_root: Path,
    run_id: Optional[str],
) -> Path:
    """Find the smoothed-profile CSV for a given (well, campaign, smoothing)."""
    folder = project_root / "data" / "processed" / "sec" / campaign / smoothing
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Processed folder not found: {folder}. Run the preprocessing batch first."
        )

    candidates = sorted(folder.glob(f"{well_id}_*__{smoothing}-*.csv"))
    if run_id is not None:
        # Cross-reference runs.csv to find the file written by that run_id.
        ledger = project_root / "results" / "runs.csv"
        if ledger.exists():
            ldf = pd.read_csv(ledger)
            row = ldf[ldf["run_id"] == run_id]
            if len(row):
                f = Path(row.iloc[0]["output_file"])
                f = f if f.is_absolute() else (project_root / f)
                if f.exists():
                    return f
        # Fall through to glob filter
        candidates = [c for c in candidates if run_id in c.name]
    if not candidates:
        msg = f"No smoothed CSV for {well_id} ({smoothing}) in {folder}"
        if run_id is not None:
            msg += f" matching run_id={run_id}"
        raise FileNotFoundError(msg)
    # Most recent wins.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_bp_json(
    well_id: str,
    campaign: str,
    smoothing: str,
    project_root: Path,
    run_id: Optional[str],
) -> Path:
    """Find the BIC-sweep JSON for a given (well, campaign, smoothing)."""
    folder = project_root / "data" / "breakpoints" / campaign
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Breakpoints folder not found: {folder}. Run the breakpoints batch first."
        )

    candidates = sorted(folder.glob(f"{well_id}_*__bp-{smoothing}-*.json"))
    if run_id is not None:
        ledger = project_root / "results" / "runs.csv"
        if ledger.exists():
            ldf = pd.read_csv(ledger)
            row = ldf[ldf["run_id"] == run_id]
            if len(row):
                f = Path(row.iloc[0]["output_file"])
                f = f if f.is_absolute() else (project_root / f)
                if f.exists():
                    return f
        candidates = [c for c in candidates if run_id in c.name]
    if not candidates:
        msg = f"No breakpoints JSON for {well_id} ({smoothing}) in {folder}"
        if run_id is not None:
            msg += f" matching run_id={run_id}"
        raise FileNotFoundError(msg)
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────
def list_available_runs(
    *,
    campaign: Optional[str] = None,
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """List smoothed profiles + breakpoint sweeps available on disk.

    Useful from an external project to check what's ready to plot
    without having to walk the filesystem manually.

    Returns
    -------
    pd.DataFrame
        Columns:
            campaign, well_id, date, smoothing,
            has_processed_csv, has_breakpoints_json,
            processed_csv_path, breakpoints_json_path,
            max_n_breakpoints (None if no JSON)
    """
    root = _project_root(project_root)
    rows: list[dict] = []

    # Walk processed CSVs
    proc_root = root / "data" / "processed" / "sec"
    if proc_root.is_dir():
        for camp_dir in sorted(proc_root.iterdir()):
            if not camp_dir.is_dir():
                continue
            if campaign is not None and camp_dir.name != campaign:
                continue
            for smooth_dir in sorted(camp_dir.iterdir()):
                if not smooth_dir.is_dir():
                    continue
                smoothing = smooth_dir.name
                for csv in sorted(smooth_dir.glob("*.csv")):
                    # Filename: {well_id}_{date}__{sig}.csv
                    stem = csv.stem
                    if "__" not in stem:
                        continue
                    head, _ = stem.split("__", 1)
                    parts = head.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    well_id, date = parts
                    rows.append({
                        "campaign": camp_dir.name,
                        "well_id": well_id,
                        "date": date,
                        "smoothing": smoothing,
                        "has_processed_csv": True,
                        "processed_csv_path": str(csv.relative_to(root)),
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "campaign", "well_id", "date", "smoothing",
            "has_processed_csv", "has_breakpoints_json",
            "processed_csv_path", "breakpoints_json_path",
            "max_n_breakpoints",
        ])

    out = pd.DataFrame(rows)

    # Now augment each row with breakpoint-JSON info if available.
    bp_paths: list[Optional[str]] = []
    bp_max_n: list[Optional[int]] = []
    for _, r in out.iterrows():
        try:
            j = _resolve_bp_json(r["well_id"], r["campaign"], r["smoothing"],
                                 root, run_id=None)
            bp_paths.append(str(j.relative_to(root)))
            try:
                with open(j) as f:
                    data = json.load(f)
                trial = next(iter(data))
                df = pd.DataFrame(data[trial]["df"])
                bp_max_n.append(int(df["n_breakpoints"].max())
                                if "n_breakpoints" in df.columns else None)
            except Exception:
                bp_max_n.append(None)
        except FileNotFoundError:
            bp_paths.append(None)
            bp_max_n.append(None)

    out["breakpoints_json_path"] = bp_paths
    out["has_breakpoints_json"] = [p is not None for p in bp_paths]
    out["max_n_breakpoints"] = bp_max_n

    return out[[
        "campaign", "well_id", "date", "smoothing",
        "has_processed_csv", "has_breakpoints_json",
        "processed_csv_path", "breakpoints_json_path",
        "max_n_breakpoints",
    ]].sort_values(["campaign", "well_id", "smoothing"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
#  Profile loader
# ─────────────────────────────────────────────────────────────────────────
def load_sec_profile(
    *,
    well_id: str,
    campaign: str,
    smoothing: str,
    run_id: Optional[str] = None,
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load a smoothed SEC profile, in both datums.

    Parameters
    ----------
    well_id : str
        e.g. ``"AW6D"``.
    campaign : str
        e.g. ``"2022_02"``.
    smoothing : {"savgol", "lowess"}
    run_id : str, optional
        Pin to a specific run via ``results/runs.csv``. If None, the
        most recently modified file wins.
    project_root : Path or str, optional
        Root of the karst_analysis project. Defaults to ``Path.cwd()``.

    Returns
    -------
    pd.DataFrame
        Always contains: ``depth_m`` (water-table datum) and
        ``sec_uS_cm``. Adds ``log10_sec_uS_cm`` if it was computed in
        the pipeline. Adds ``depth_bgl_m`` (ground-level datum) if a
        vadose value was found in ``wells.csv`` (already present from
        the pipeline; otherwise computed here on the fly).

        Metadata columns appended for traceability:
            well_id, campaign, smoothing, source_file
    """
    _validate_smoothing(smoothing)
    root = _project_root(project_root)
    csv_path = _resolve_processed_csv(well_id, campaign, smoothing, root, run_id)

    df = pd.read_csv(csv_path)

    # If depth_bgl_m wasn't written by the pipeline, try to derive it now.
    if "depth_bgl_m" not in df.columns and "depth_m" in df.columns:
        v = _vadose_for(well_id, root)
        if v is not None:
            df["depth_bgl_m"] = df["depth_m"].astype(float) + v

    df["well_id"] = well_id
    df["campaign"] = campaign
    df["smoothing"] = smoothing
    df["source_file"] = str(csv_path.relative_to(root))
    return df


# ─────────────────────────────────────────────────────────────────────────
#  Breakpoint loader at a given N
# ─────────────────────────────────────────────────────────────────────────
def load_breakpoints_at_n(
    *,
    well_id: str,
    campaign: str,
    smoothing: str,
    n: int,
    trial: str = "trial_1",
    run_id: Optional[str] = None,
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load breakpoints for a specific N, ready to plot.

    Parameters
    ----------
    well_id, campaign, smoothing : same as :func:`load_sec_profile`.
    n : int
        How many breakpoints to extract from the BIC sweep. Must be
        within the range used by the original sweep.
    trial : str, default "trial_1"
        Which trial to read. Special values:
            - ``"best_bic"`` : pick the trial with lowest BIC at this N.
            - ``"trial_1"``, ``"trial_2"``, ... : explicit trial.
    run_id : str, optional
        Pin to a specific run.
    project_root : Path or str, optional

    Returns
    -------
    pd.DataFrame
        One row per breakpoint, ordered by depth ascending. Columns:
            bp_index            : 1, 2, ..., n
            depth_m             : breakpoint depth, water-table datum
            depth_bgl_m         : same, ground-level datum (NaN if unknown)
            ci_lower_m          : 95% CI lower bound, water-table datum
            ci_upper_m          : 95% CI upper bound, water-table datum
            ci_lower_bgl_m      : same, ground-level datum (NaN if unknown)
            ci_upper_bgl_m      : same, ground-level datum (NaN if unknown)
            ci_width_m          : ci_upper - ci_lower (datum-invariant)
            sec_at_bp_log10     : fitted y-value (log10 µS/cm) at the BP
            sec_at_bp_uS_cm     : same value converted to linear µS/cm
                                  (10**sec_at_bp_log10). Use this for
                                  plotting on a linear x-axis.
            n                   : the N requested
            bic, rss, converged : global metrics for this N
            well_id, campaign, smoothing, trial, source_file : metadata
    """
    _validate_smoothing(smoothing)
    if n < 1:
        raise ValueError(f"n must be ≥ 1; got {n}.")

    root = _project_root(project_root)
    json_path = _resolve_bp_json(well_id, campaign, smoothing, root, run_id)
    bic_data = load_bic_json(json_path)

    if trial == "best_bic":
        trial = _select_best_trial_for_n(bic_data, n)
    if trial not in bic_data:
        raise KeyError(
            f"Trial '{trial}' not found in {json_path.name}. "
            f"Available: {list(bic_data.keys())}"
        )

    df = pd.DataFrame(bic_data[trial]["df"])
    if "n_breakpoints" not in df.columns:
        raise ValueError(f"'n_breakpoints' missing from {json_path}")

    match = df[df["n_breakpoints"] == n]
    if not len(match):
        available = sorted(df["n_breakpoints"].unique().tolist())
        raise ValueError(
            f"N={n} not in JSON. Available: {available}. JSON: {json_path.name}"
        )
    row = match.iloc[0]
    estimates = row["estimates"]
    if isinstance(estimates, str):
        estimates = json.loads(estimates)

    # Pull global fit metrics for this N.
    bic_val      = float(row["bic"])      if "bic"      in df.columns and pd.notna(row.get("bic")) else np.nan
    rss_val      = float(row["rss"])      if "rss"      in df.columns and pd.notna(row.get("rss")) else np.nan
    converged    = bool(row["converged"]) if "converged" in df.columns else None

    # Pull line params (for sec_at_bp).
    def _est(k):
        v = estimates.get(k)
        if isinstance(v, dict):
            return float(v.get("estimate"))
        return float(v) if v is not None else None

    const  = _est("const")
    alpha1 = _est("alpha1")
    betas  = [_est(f"beta{i}") for i in range(1, n + 1)]

    rows = []
    for i in range(1, n + 1):
        bp_entry = estimates.get(f"breakpoint{i}")
        if bp_entry is None:
            continue
        if isinstance(bp_entry, dict):
            x_bp = float(bp_entry["estimate"])
            ci   = bp_entry.get("confidence_interval")
            if ci is None:
                ci_lo, ci_hi = np.nan, np.nan
            else:
                ci_lo, ci_hi = float(ci[0]), float(ci[1])
        else:
            x_bp = float(bp_entry)
            ci_lo, ci_hi = np.nan, np.nan

        # Predict y at the breakpoint using the piecewise-linear formula.
        if const is None or alpha1 is None:
            y_at_bp = np.nan
        else:
            other_bps = [
                _est(f"breakpoint{j}") for j in range(1, n + 1) if j != i
            ]
            y_at_bp = const + alpha1 * x_bp
            for b, bp_j in zip(betas, [_est(f"breakpoint{j}") for j in range(1, n + 1)]):
                if bp_j is not None and x_bp > bp_j and b is not None:
                    y_at_bp += b * (x_bp - bp_j)

        rows.append({
            "bp_index":         i,
            "depth_m":          x_bp,
            "ci_lower_m":       ci_lo,
            "ci_upper_m":       ci_hi,
            "ci_width_m":       (ci_hi - ci_lo) if pd.notna(ci_lo) and pd.notna(ci_hi) else np.nan,
            "sec_at_bp_log10":  y_at_bp,
            "sec_at_bp_uS_cm":  10.0 ** y_at_bp if pd.notna(y_at_bp) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("depth_m").reset_index(drop=True)
    # Re-index bp_index after sorting by depth so BP1 is shallowest.
    out["bp_index"] = out.index + 1

    # Datum conversion to BGL.
    vadose = _vadose_for(well_id, root)
    if vadose is not None:
        out["depth_bgl_m"]     = out["depth_m"]    + vadose
        out["ci_lower_bgl_m"]  = out["ci_lower_m"] + vadose
        out["ci_upper_bgl_m"]  = out["ci_upper_m"] + vadose
    else:
        out["depth_bgl_m"]     = np.nan
        out["ci_lower_bgl_m"]  = np.nan
        out["ci_upper_bgl_m"]  = np.nan

    # Metadata + global metrics
    out["n"]           = n
    out["bic"]         = bic_val
    out["rss"]         = rss_val
    out["converged"]   = converged
    out["well_id"]     = well_id
    out["campaign"]    = campaign
    out["smoothing"]   = smoothing
    out["trial"]       = trial
    out["vadose_thickness_m"] = vadose if vadose is not None else np.nan
    out["source_file"] = str(json_path.relative_to(root))

    # Final column order
    return out[[
        "bp_index",
        "depth_m", "depth_bgl_m",
        "ci_lower_m", "ci_upper_m",
        "ci_lower_bgl_m", "ci_upper_bgl_m",
        "ci_width_m",
        "sec_at_bp_log10", "sec_at_bp_uS_cm",
        "n", "bic", "rss", "converged",
        "well_id", "campaign", "smoothing", "trial",
        "vadose_thickness_m", "source_file",
    ]]


# ─────────────────────────────────────────────────────────────────────────
#  BIC curve loader
# ─────────────────────────────────────────────────────────────────────────
def load_bic_curve(
    *,
    well_id: str,
    campaign: str,
    smoothing: str,
    trial: str = "trial_1",
    run_id: Optional[str] = None,
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load the BIC curve (and RSS) over N for a given run.

    Useful in the external project for plotting an "elbow" or
    "knee" diagnostic alongside the breakpoints, helping the user
    decide which N is most plausible.

    Returns
    -------
    pd.DataFrame
        Columns: n_breakpoints, bic, rss, converged, well_id, campaign,
        smoothing, trial, source_file.
    """
    _validate_smoothing(smoothing)
    root = _project_root(project_root)
    json_path = _resolve_bp_json(well_id, campaign, smoothing, root, run_id)
    bic_data = load_bic_json(json_path)

    if trial not in bic_data:
        raise KeyError(
            f"Trial '{trial}' not in JSON. Available: {list(bic_data.keys())}"
        )

    df = pd.DataFrame(bic_data[trial]["df"])
    keep = ["n_breakpoints"]
    for c in ("bic", "rss", "converged"):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    out["well_id"]     = well_id
    out["campaign"]    = campaign
    out["smoothing"]   = smoothing
    out["trial"]       = trial
    out["source_file"] = str(json_path.relative_to(root))
    return out.sort_values("n_breakpoints").reset_index(drop=True)
