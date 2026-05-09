"""Breakpoint detection for ERT 1D profiles, with seed discovery.

Why a separate module from SEC
-------------------------------
The SEC breakpoint helpers in
``karst_analysis.sec.breakpoints`` use ``piecewise_regression`` without
managing the global numpy RNG state, which means a converged fit is
not bit-for-bit reproducible across runs. For ERT we want explicit
reproducibility (the fit's converged seed is recorded alongside the
breakpoints), so this module wraps ``pw.Fit`` directly and iterates
seeds until one converges.

What we DO reuse from SEC
-------------------------
``extract_breakpoints(fit) -> pd.DataFrame`` from
``karst_analysis.sec.breakpoints`` is fully generic (it just reads the
``Fit`` object and emits a tidy DataFrame). We import and reuse it.

Method
------
``piecewise_regression`` does not expose a ``seed`` parameter. Empirical
testing showed that calling ``np.random.seed(K)`` IMMEDIATELY before
``pw.Fit(...)`` makes the fit deterministic, because under the hood
``pw`` uses the global ``np.random`` state.

Pseudo-code:

    for s in [start_seed, start_seed+1, ...]:
        np.random.seed(s)
        fit = pw.Fit(x, y, n_breakpoints=N, ...)
        if fit.best_muggeo is not None:        # converged
            return fit, s
    raise RuntimeError("no seed converged within max_seed_attempts")

The first converged seed wins — see the entry on multi-seed BIC
comparison in NOTES_open_questions.md (deferred to post-thesis).

Reproducibility caveat
----------------------
The seed → result mapping holds within a single Python process and a
fixed numpy version. Across numpy versions the global RNG algorithm
may change, breaking exact reproducibility (BIC differences should be
negligible but bit-for-bit identity is not guaranteed). This is the
same caveat the SEC pipeline carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import piecewise_regression as pw

from karst_analysis.sec.breakpoints import extract_breakpoints


@dataclass(frozen=True)
class ErtBreakpointFit:
    """Result of a successful ERT breakpoint detection.

    Attributes
    ----------
    breakpoints : pd.DataFrame
        Output of ``extract_breakpoints(fit)`` — one row per
        breakpoint with positions and confidence intervals.
    n_breakpoints : int
        Number of breakpoints requested (and found).
    seed_used : int
        The numpy seed that produced this converged fit.
    seeds_tried : tuple[int, ...]
        Every seed tried before (and including) the one that
        converged. Length 1 means the first seed worked.
    bic : float
        Bayesian Information Criterion of the converged fit.
    rss : float
        Residual sum of squares of the converged fit.
    """

    breakpoints: pd.DataFrame
    n_breakpoints: int
    seed_used: int
    seeds_tried: tuple[int, ...]
    bic: float
    rss: float


def detect_breakpoints_with_seed_discovery(
    x: np.ndarray,
    y: np.ndarray,
    n_breakpoints: int,
    *,
    max_seed_attempts: int = 20,
    start_seed: int = 0,
    tolerance: float = 1e-5,
    min_distance: float = 0.01,
) -> ErtBreakpointFit:
    """Find breakpoints with deterministic seed discovery.

    Iterates numpy seeds starting at ``start_seed``, calling
    ``pw.Fit(x, y, n_breakpoints=n_breakpoints, ...)`` after seeding
    the global RNG. Returns the first converged fit.

    Parameters
    ----------
    x, y : ndarray
        Profile coordinates. For ERT, callers pass
        ``(depth_bgl_m, resistlog10)`` — detection is done in log10
        space by convention.
    n_breakpoints : int
        Number of breakpoints to fit.
    max_seed_attempts : int, default 20
        Maximum consecutive seeds to try before giving up.
    start_seed : int, default 0
        First seed to try. Subsequent seeds are start_seed+1,
        start_seed+2, ...
    tolerance, min_distance : float
        Passed straight to ``pw.Fit``. Defaults match the SEC pipeline.

    Returns
    -------
    ErtBreakpointFit

    Raises
    ------
    RuntimeError
        If no seed in [start_seed, start_seed+max_seed_attempts) yields
        a converged fit. The error message lists the seeds tried so the
        caller can adjust ``start_seed`` / ``max_seed_attempts`` or
        reduce ``n_breakpoints``.

    Notes
    -----
    A "converged fit" means ``fit.best_muggeo is not None``. When
    convergence fails, ``pw.Fit`` does not raise — it just leaves
    ``best_muggeo = None`` — so we must check explicitly.

    TODO (post-thesis, NOTES_open_questions.md #5b): extend to find
    the first 3 converged seeds, compare their BIC, and keep the
    best-BIC one. The current implementation keeps only the first.
    """
    if n_breakpoints < 1:
        raise ValueError(f"n_breakpoints must be >= 1, got {n_breakpoints}")
    if max_seed_attempts < 1:
        raise ValueError(
            f"max_seed_attempts must be >= 1, got {max_seed_attempts}"
        )

    seeds_tried: list[int] = []
    for offset in range(max_seed_attempts):
        seed = start_seed + offset
        seeds_tried.append(seed)

        np.random.seed(seed)
        fit = pw.Fit(
            x, y,
            n_breakpoints=n_breakpoints,
            tolerance=tolerance,
            min_distance_between_breakpoints=min_distance,
        )
        if fit.best_muggeo is None:
            continue  # try next seed

        # Converged.
        bps = extract_breakpoints(fit)
        # Pull BIC and RSS from the Muggeo result.
        muggeo = fit.best_muggeo.best_fit
        bic = float(muggeo.bic)
        rss = float(muggeo.residual_sum_squares)

        return ErtBreakpointFit(
            breakpoints=bps,
            n_breakpoints=n_breakpoints,
            seed_used=seed,
            seeds_tried=tuple(seeds_tried),
            bic=bic,
            rss=rss,
        )

    raise RuntimeError(
        f"No converged fit found for n_breakpoints={n_breakpoints} "
        f"after trying seeds {seeds_tried[0]}..{seeds_tried[-1]} "
        f"({len(seeds_tried)} attempts). Try increasing "
        f"max_seed_attempts, changing start_seed, or reducing "
        f"n_breakpoints."
    )
