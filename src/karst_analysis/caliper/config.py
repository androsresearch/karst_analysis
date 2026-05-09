"""Caliper-pipeline configuration constants.

This module is the single source of truth for the parameters that drive
the caliper baseline / detection / severity pipeline. All other modules
in ``karst_analysis.caliper`` import from here.

Why a Python module and not a YAML file
---------------------------------------
The constants below are scientific decisions, not user preferences:

    * ``OFFSET_CM = 1.6`` is justified by the variance decomposition of
      caliper transducer noise (AW5O smooth sonic-cored hole) plus
      rotary-auger drilling roughness (AW5D vs AW5O comparison).
    * ``MILD/MODERATE_MAX_EXCESS_CM`` are multiples of OFFSET_CM
      (2× and 6× respectively) — the relationship is structural.
    * ``TRIM_DEPTHS_M`` are picked per well after looking at the upper
      drilling-disturbed zone in each log.

Editing any of these must be a deliberate scientific decision, traceable
in version control, with a comment explaining why. A YAML config would
make these changes too easy to perform without justification.

If a sensitivity analysis is needed (e.g. running the pipeline with
several values of OFFSET_CM), the recommended pattern is to import the
processing functions directly and override parameters at the call site,
not to edit this file.

Convention
----------
All depths are expressed in metres **below ground level (BGL)**, i.e.
positive numbers, with 0 = ground level. This matches the original
LAS files and the SEC sub-package. The word "elevation" is reserved
for the future case where absolute elevation (m above sea level) is
known from differential GPS.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────
#  Priority wells of the project
# ──────────────────────────────────────────────────────────────────────
PRIORITY_WELLS: list[str] = ["AW5D", "AW6D", "BW3D", "LRS69D", "LRS70D"]


# ──────────────────────────────────────────────────────────────────────
#  Per-well trim depths, in BGL-positive metres.
#  Boundary between "shallow drilling-disturbed" and "deep" sub-zones
#  for the cumulative-min split fit.
# ──────────────────────────────────────────────────────────────────────
TRIM_DEPTHS_M: dict[str, float] = {
    "AW5D":   5.0,
    "AW6D":   5.0,
    "BW3D":   7.0,
    "LRS69D": 7.0,
    "LRS70D": 5.0,
}


# ──────────────────────────────────────────────────────────────────────
#  Detection rule:
#      C(z) > B(z) + OFFSET_CM + K_SIGMA * sigma_inst_cm
# ──────────────────────────────────────────────────────────────────────
OFFSET_CM:        float = 1.6     # fixed additive offset over baseline
K_SIGMA:          float = 1.0     # multiplier on the instrumental noise term
L_MIN_M:          float = 0.06    # minimum run length to keep a zone
SATURATION_CM:    float = 32.50   # caliper saturates at this aperture


# ──────────────────────────────────────────────────────────────────────
#  Severity binning (excess measured FROM THE THRESHOLD = B + offset)
# ──────────────────────────────────────────────────────────────────────
MILD_MAX_EXCESS_CM:     float = 2.0 * OFFSET_CM   # 3.2 cm
MODERATE_MAX_EXCESS_CM: float = 6.0 * OFFSET_CM   # 9.6 cm


# ──────────────────────────────────────────────────────────────────────
#  Noise-estimation defaults (used by karst_analysis.caliper.noise)
# ──────────────────────────────────────────────────────────────────────
DETREND_WINDOW_M: float = 0.30

# Reference intervals used to estimate sigma_inst, in BGL-positive metres.
#     AW5O: smooth sonic-cored hole — used as the "transducer-only" noise
#     AW5D: rotary-auger hole — used to compute the drilling contribution
NOISE_INTERVAL_AW5O: tuple[float, float] = (1.0, 5.0)
NOISE_INTERVAL_AW5D: tuple[float, float] = (15.0, 20.0)


# ──────────────────────────────────────────────────────────────────────
#  Default baseline-fit options
# ──────────────────────────────────────────────────────────────────────
DEFAULT_INTERP_KIND: str = "linear"   # 'step', 'linear', 'pchip'
DEFAULT_DIRECTION:   str = "top_down" # 'top_down', 'bottom_up'
DEFAULT_IQR_K:       float = 1.5      # Tukey lower-fence multiplier
