"""Video-log notes loader and parser.

Submodules
----------
parsing : depth-token + typo-fix helpers
io      : xlsx sheet loader

The video-log xlsx is hand-typed during fieldwork; the parser tolerates
several depth formats (``"1.5"``, ``"1.5 m"``, ``"1.5-2.0"`` with hyphen
or en-dash) and applies a curated typo-fix list.
"""

from karst_analysis.videolog.io import load_video_notes, DEFAULT_VIDEOLOG_XLSX
from karst_analysis.videolog.parsing import (
    apply_typo_fixes, parse_depth_token, TYPO_FIXES,
)

__all__ = [
    "load_video_notes",
    "DEFAULT_VIDEOLOG_XLSX",
    "apply_typo_fixes",
    "parse_depth_token",
    "TYPO_FIXES",
]
