"""Specific Electrical Conductivity (SEC) profile analysis.

This sub-package contains the active analysis pipeline for SEC profiles
collected with YSI probes in monitoring wells.

Sub-modules:
    io            : SEC-specific column conventions and loaders.
    preprocessing : cleaning, smoothing (savgol/lowess), pipelines.
    breakpoints   : segmented regression and BIC-based selection.
    viz           : diagnostic and comparison plots.
"""
