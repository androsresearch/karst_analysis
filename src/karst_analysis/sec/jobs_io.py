"""Shared parser for `slopes_jobs_*.yml` job files.

This module owns the canonical representation of a "slopes job" — a
single (well, method, trial, N) combination chosen by the user after
inspecting the breakpoint figures. Multiple scripts and panels consume
the same YAML, so the parser lives here rather than being duplicated.

Consumers
---------
* ``scripts/slopes_batch.py``         — computes chord slopes per job.
* ``scripts/sec_caliper_video_panels.py``
                                      — renders the three-technique
                                        panel for each job, matching
                                        the trial/N/method choices.

YAML schema
-----------
::

    campaign: "2022_02"
    # Optional global default for BOT-MZ SEC threshold (µS/cm).
    bot_mz_sec_threshold: 40000
    # Optional global default for TOP-MZ SEC threshold (µS/cm).
    # Omit to disable the constraint (legacy behaviour: pure curvature).
    top_mz_sec_threshold: 10000

    jobs:
      - well: LRS70D
        method: lowess        # one of {"savgol", "lowess"}
        trial: trial_3        # "trial_1", "trial_2", ... or "best_bic"
        n: 15                 # number of breakpoints
        bot_mz_sec_threshold: 40000   # optional per-job override
        top_mz_sec_threshold: 10000   # optional per-job override

Backwards compatibility
-----------------------
PyYAML interprets unquoted ``2022_02`` as the integer ``202202`` (Python
allows underscores as digit separators). ``_normalise_campaign`` accepts
both and warns when the YAML value was unquoted.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    """One unit of work: a (well, method, trial, N) combination.

    Attributes
    ----------
    well : str
        Well ID (e.g. ``"LRS70D"``).
    method : str
        Smoothing method — one of ``{"savgol", "lowess"}``.
    trial : str
        Trial identifier — ``"trial_1"``, ``"trial_2"``, ... or
        ``"best_bic"`` to auto-select.
    n : int
        Number of breakpoints (must be in the BIC sweep range).
    bot_mz_sec_threshold : float, optional
        Per-job override for the BOT-MZ SEC threshold (µS/cm). ``None``
        means "fall back to the YAML default, then to the
        ``compute_slopes`` built-in default".
    top_mz_sec_threshold : float, optional
        Per-job override for the TOP-MZ SEC threshold (µS/cm). ``None``
        means "fall back to the YAML default, then to disabled
        (legacy pure-curvature behaviour)".
    """
    well: str
    method: str
    trial: str
    n: int
    bot_mz_sec_threshold: Optional[float] = None
    top_mz_sec_threshold: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────
def trial_index(trial_name: str) -> int:
    """Map ``"trial_3"`` → ``3``. Falls back to ``1`` if no digits.

    Used to build filename suffixes like ``__t3``.
    """
    m = re.search(r"(\d+)$", trial_name)
    return int(m.group(1)) if m else 1


def _normalise_campaign(value) -> str:
    """Coerce a campaign value to canonical 'YYYY_MM' string.

    PyYAML interprets unquoted ``2022_02`` as the integer ``202202``
    (Python allows underscores as digit separators). To stay
    backwards-friendly we accept both: if we receive a 6-digit int, we
    reformat it as ``YYYY_MM`` and warn the user.
    """
    if isinstance(value, int):
        s = str(value)
        if len(s) == 6 and s[:4].isdigit() and s[4:].isdigit():
            recovered = f"{s[:4]}_{s[4:]}"
            logger.warning(
                "campaign was parsed as integer %d (the YAML value "
                "likely lacked quotes). Reconstructed as '%s'. "
                "Consider quoting it in the YAML: campaign: \"%s\"",
                value, recovered, recovered,
            )
            return recovered
        return s
    if isinstance(value, str):
        return value
    raise ValueError(
        f"campaign must be a string (e.g. '2022_02'); "
        f"got {type(value).__name__}: {value!r}"
    )


def load_jobs_file(
    path: Path | str,
) -> tuple[str, Optional[float], Optional[float], list[Job]]:
    """Parse a YAML jobs file.

    Parameters
    ----------
    path : Path or str
        Path to the YAML jobs file.

    Returns
    -------
    (campaign, default_bot_threshold, default_top_threshold, jobs)
        - ``campaign`` : canonical ``'YYYY_MM'`` string.
        - ``default_bot_threshold`` : optional global default for
          ``bot_mz_sec_threshold``; ``None`` if absent at YAML root.
        - ``default_top_threshold`` : optional global default for
          ``top_mz_sec_threshold``; ``None`` if absent at YAML root.
        - ``jobs`` : list of :class:`Job` instances. Per-job thresholds
          are on ``bot_mz_sec_threshold`` / ``top_mz_sec_threshold`` if
          specified, else ``None``.

    Raises
    ------
    ValueError
        If the YAML is malformed, missing required keys, or has an
        invalid method.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{path} must be a YAML mapping with 'campaign' and 'jobs'."
        )

    raw_campaign = cfg.get("campaign")
    if raw_campaign is None:
        raise ValueError(f"{path}: missing 'campaign' key.")
    campaign = _normalise_campaign(raw_campaign)

    default_bot_threshold = cfg.get("bot_mz_sec_threshold")
    if default_bot_threshold is not None:
        default_bot_threshold = float(default_bot_threshold)

    default_top_threshold = cfg.get("top_mz_sec_threshold")
    if default_top_threshold is not None:
        default_top_threshold = float(default_top_threshold)

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
                f"{path} job #{i}: method must be 'savgol' or 'lowess'; "
                f"got {j['method']!r}."
            )
        per_job_bot = j.get("bot_mz_sec_threshold")
        if per_job_bot is not None:
            per_job_bot = float(per_job_bot)
        per_job_top = j.get("top_mz_sec_threshold")
        if per_job_top is not None:
            per_job_top = float(per_job_top)
        jobs.append(Job(
            well=str(j["well"]),
            method=str(j["method"]),
            trial=str(j["trial"]),
            n=int(j["n"]),
            bot_mz_sec_threshold=per_job_bot,
            top_mz_sec_threshold=per_job_top,
        ))

    return campaign, default_bot_threshold, default_top_threshold, jobs
