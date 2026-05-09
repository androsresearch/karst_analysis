"""Shared IO utilities used across techniques.

Currently exposes:
    parse_well_filename  : decode metadata from a CSV filename.
    resolve_figure_dir   : v13 figure-output path resolver.
    FIGURES_ROOT         : the project-wide root for figure outputs.

Technique-specific loaders live in their own sub-packages
(e.g. ``karst_analysis.sec.io``).
"""

from karst_analysis.io.filenames import parse_well_filename
from karst_analysis.io.figure_paths import (
    FIGURES_ROOT,
    campaign_subdir_label,
    resolve_figure_dir,
)

__all__ = [
    "parse_well_filename",
    "FIGURES_ROOT",
    "campaign_subdir_label",
    "resolve_figure_dir",
]
