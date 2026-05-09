"""Run tracking for reproducibility across iterative analysis.

The pre-processing → breakpoints workflow is iterative: smooth a profile,
detect breakpoints, evaluate visually, change parameters, re-run. Without
discipline, the proliferation of intermediate files becomes impossible
to track.

This module provides three layers of traceability:

1. **Method signature**  — a short human-readable string that encodes the
   parameter set, e.g. ``"lowess-f0.05-i2-pava"`` or ``"savgol-w11-o3-seg"``.
   Embedded in output filenames so a quick look at a folder reveals which
   parameter set produced each file.

2. **Run ID**  — first 8 characters of the SHA-1 hash of the canonicalised
   parameter dict. Stable across runs (same params → same id), so runs can
   be referenced unambiguously in notes, notebooks, and the thesis.

3. **runs.csv**  — a single-file ledger. Every run appends one row recording
   the timestamp, well, stage, parameters, input file, output file, run_id,
   and free-form notes. Filterable in Excel; queryable from pandas.

Usage
-----
>>> from karst_analysis.runs import register_run
>>> with register_run(
...     stage="smoothing",
...     well_id="AW6D",
...     date="20220215",
...     input_file="data/raw/sec/2022_02/AW6_D_YSI_20220215.csv",
...     params={"method": "lowess", "frac": 0.05, "iter": 2, "pava": True},
...     output_dir="data/processed/sec/2022_02/lowess",
... ) as run:
...     # Run.output_path is auto-generated and includes the signature.
...     df_processed = process_lowess(df_raw, frac=0.05)
...     df_processed.to_csv(run.output_path, index=False)
...     run.note = "frac=0.05 quita ruido pero conserva transiciones"
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd


# Default ledger location, relative to the project root.
DEFAULT_RUNS_LEDGER = Path("results/runs.csv")

# Order of columns in the ledger — ensures consistent CSV layout across
# runs even if dict iteration order differs.
LEDGER_COLUMNS = [
    "run_id",
    "timestamp",
    "stage",
    "well_id",
    "date",
    "method_signature",
    "input_file",
    "output_file",
    "params_json",
    "notes",
]


# ─────────────────────────────────────────────────────────────────────────
#  Hashing and signature helpers
# ─────────────────────────────────────────────────────────────────────────
def _canonical_json(params: dict[str, Any]) -> str:
    """Return a deterministic JSON string for hashing.

    Sorts keys and uses compact separators so the same dict always
    produces the same bytes regardless of insertion order.
    """
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def generate_run_id(params: dict[str, Any]) -> str:
    """Hash the parameter dict to an 8-char ID.

    Same parameters always yield the same ID (idempotent). Different
    parameters yield different IDs with overwhelming probability.

    Parameters
    ----------
    params : dict
        Parameter dictionary. Must be JSON-serialisable.

    Returns
    -------
    str
        8-character hexadecimal hash.
    """
    blob = _canonical_json(params).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def build_method_signature(params: dict[str, Any]) -> str:
    """Build a short human-readable signature from a params dict.

    The signature is what you read on disk when scanning a folder. It is
    NOT meant to be reversible — for that, use the run_id and look up
    full params in runs.csv.

    Conventions:
        method=savgol → "savgol-w{window}-o{order}[-seg]"
        method=lowess → "lowess-f{frac}-i{iter}[-pava]"
        otherwise     → "method-key1{val1}-key2{val2}-..."

    Parameters
    ----------
    params : dict
        Must contain at least a ``method`` key.

    Returns
    -------
    str
        Signature string, safe for use in filenames (no slashes, spaces,
        or special chars).
    """
    method = params.get("method", "unknown")

    if method == "savgol":
        window = params.get("window", "?")
        order = params.get("order", "?")
        seg = "-seg" if params.get("segmented", False) else ""
        return f"savgol-w{window}-o{order}{seg}"

    if method == "lowess":
        frac = params.get("frac", "?")
        n_iter = params.get("iter", "?")
        pava = "-pava" if params.get("pava", False) else ""
        return f"lowess-f{frac}-i{n_iter}{pava}"

    if method == "breakpoints":
        max_bp = params.get("max_breakpoints", "?")
        n_trials = params.get("n_trials", "?")
        # Include smoothing_method so files for savgol vs lowess do NOT collide.
        smooth = params.get("smoothing_method")
        smooth_part = f"-{smooth}" if smooth else ""
        return f"bp{smooth_part}-max{max_bp}-t{n_trials}"

    if method == "convergence_sec_caliper":
        # The run_tag is the version-bumpable identity; appended params
        # make the signature differentiate ad-hoc sensitivity runs.
        tag = params.get("run_tag", "convergence_sec_caliper")
        rule = params.get("matching_rule", "?")
        tol = params.get("tolerance_m", "?")
        amin = params.get("sec_agreement_min", "?")
        return f"{tag}-{rule}-tol{tol}-amin{amin}"

    # Generic fallback — alphabetical order for stability.
    parts = [method]
    for k in sorted(params):
        if k == "method":
            continue
        v = params[k]
        if isinstance(v, bool):
            if v:
                parts.append(k)
        else:
            parts.append(f"{k}{v}")
    return "-".join(str(p) for p in parts)


def build_output_filename(
    well_id: str,
    date: str,
    method_signature: str,
    extension: str = "csv",
) -> str:
    """Compose a self-describing output filename.

    Format: ``{well_id}_{date}__{method_signature}.{extension}``

    The double underscore separates the well identity from the method
    metadata, making the filename easy to parse visually.

    Examples
    --------
    >>> build_output_filename("AW6D", "20220215", "lowess-f0.05-i2-pava")
    'AW6D_20220215__lowess-f0.05-i2-pava.csv'
    """
    return f"{well_id}_{date}__{method_signature}.{extension}"


# ─────────────────────────────────────────────────────────────────────────
#  Run object
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Run:
    """A single recorded analysis step.

    Created by :func:`register_run`. Within the ``with`` block, the user
    code does the actual work and writes outputs to ``self.output_path``.
    The note attribute can be set freely. On context exit the run is
    appended to the ledger.
    """
    run_id: str
    timestamp: str
    stage: str
    well_id: str
    date: str
    method_signature: str
    input_file: str
    output_file: str
    params: dict[str, Any]
    output_path: Path
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_ledger_row(self) -> dict[str, str]:
        """Render the run as a flat dict ready for the CSV ledger."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "well_id": self.well_id,
            "date": self.date,
            "method_signature": self.method_signature,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "params_json": _canonical_json(self.params),
            "notes": self.note,
        }


# ─────────────────────────────────────────────────────────────────────────
#  Ledger I/O
# ─────────────────────────────────────────────────────────────────────────
def _ensure_ledger(path: Path) -> None:
    """Create the ledger file with headers if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(path, index=False)


def append_to_ledger(run: Run, ledger_path: Optional[Path] = None) -> None:
    """Append a single run as a row in the ledger CSV."""
    path = Path(ledger_path) if ledger_path is not None else DEFAULT_RUNS_LEDGER
    _ensure_ledger(path)

    row = run.to_ledger_row()
    pd.DataFrame([row], columns=LEDGER_COLUMNS).to_csv(
        path, mode="a", header=False, index=False
    )


def read_ledger(ledger_path: Optional[Path] = None) -> pd.DataFrame:
    """Read the ledger as a DataFrame for filtering / inspection."""
    path = Path(ledger_path) if ledger_path is not None else DEFAULT_RUNS_LEDGER
    _ensure_ledger(path)
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────
@contextmanager
def register_run(
    *,
    stage: str,
    well_id: str,
    date: str,
    input_file: str | Path,
    params: dict[str, Any],
    output_dir: str | Path,
    extension: str = "csv",
    ledger_path: Optional[str | Path] = None,
) -> Iterator[Run]:
    """Register a run and yield a :class:`Run` object for use inside a ``with`` block.

    The function:
      1. Builds the method signature and run_id from ``params``.
      2. Composes the output filename and ensures ``output_dir`` exists.
      3. Yields a Run instance so the caller can write the output file.
      4. On exit (normal or exception), appends a row to the ledger
         only if no exception was raised. Failed runs are not recorded
         to keep the ledger clean.

    Parameters
    ----------
    stage : str
        High-level pipeline stage. Conventional values:
        ``"smoothing"``, ``"breakpoints"``, ``"breakpoints_chosen"``.
    well_id : str
        Well identifier, e.g. ``"AW6D"``.
    date : str
        Acquisition date in ``YYYYMMDD`` format.
    input_file : str or Path
        Path to the file the run reads from. Stored in the ledger for
        traceability across pipeline stages.
    params : dict
        Parameter dictionary. Must include a ``method`` key. Used both
        to build the signature/run_id and persisted in the ledger.
    output_dir : str or Path
        Directory where ``run.output_path`` will live. Created if missing.
    extension : str, default "csv"
        Output file extension (no dot).
    ledger_path : str or Path, optional
        Override the default ledger location.

    Yields
    ------
    Run
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    signature = build_method_signature(params)
    run_id = generate_run_id(params)
    output_filename = build_output_filename(well_id, date, signature, extension)
    output_path = output_dir_p / output_filename

    run = Run(
        run_id=run_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        stage=stage,
        well_id=well_id,
        date=date,
        method_signature=signature,
        input_file=str(input_file),
        output_file=str(output_path),
        params=params,
        output_path=output_path,
    )

    try:
        yield run
    except Exception:
        # Do not record failed runs in the ledger.
        raise
    else:
        append_to_ledger(run, ledger_path=Path(ledger_path) if ledger_path else None)
