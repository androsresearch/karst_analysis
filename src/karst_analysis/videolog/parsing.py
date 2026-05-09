"""Text and depth parsing helpers for video-log notes.

Separates the parsing logic from the loader so it can be tested
independently and reused by other consumers (e.g. the convergence
panel that displays Ardaman lithology entries with similar formatting).

Migration history
-----------------
v5.1: extracted from ``caliper_videolog_panel.py``. The TYPO_FIXES list
is preserved verbatim so any downstream rendering reproduces the
original cleaned-up text.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  Typo fixes for hand-typed video-log text
# ──────────────────────────────────────────────────────────────────────
TYPO_FIXES: list[tuple[str, str]] = [
    ("occuring",          "occurring"),
    ("beome",             "become"),
    ("Medium-lareg",      "Medium-large"),
    ("Eneter",            "Enter"),
    ("Botom",             "Bottom"),
    ("detriturs",         "detritus"),
    (", , ",              ", "),
    ("Meidum",            "Medium"),
    ("Smoo, ",            "Smooth, "),
    ("yellow.brown",      "yellow/brown"),
    ("beyonf",            "beyond"),
    ("salintiy",          "salinity"),
    ("largerdissolution", "larger dissolution"),
    ("Swiss chess",       "Swiss cheese"),
    ("intesifies",        "intensifies"),
    ("ocassional",        "occasional"),
    ("'sfc'",             "surface"),
    ("cloduy",            "cloudy"),
    ("Moertaely",         "Moderately"),
]


def apply_typo_fixes(text: str) -> str:
    """Apply the curated typo-fix list and trim whitespace.

    The list is intentionally bespoke (not a general spell-checker) — it
    only fixes typos actually present in the source xlsx. Adding new
    fixes is fine; removing existing ones changes the rendered output.
    """
    s = text
    for pat, rep in TYPO_FIXES:
        s = s.replace(pat, rep)
    return s.strip()


# ──────────────────────────────────────────────────────────────────────
#  Depth-token parsing
# ──────────────────────────────────────────────────────────────────────
# Match "1.5-2.0", "1.5–2.0", "1.5—2.0" (hyphen, en-dash, em-dash),
# with arbitrary whitespace.
_RANGE_RE = re.compile(
    r"^\s*(?P<a>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<b>\d+(?:\.\d+)?)\s*$"
)


def parse_depth_token(token) -> tuple[Optional[float], Optional[float]]:
    """Parse a depth cell into ``(z_top, z_bot)`` in **positive metres**.

    The video-log xlsx uses several formats for the depth column:

        * Numeric (float or int)              → returned as ``(v, v)``
        * String ``"1.5"``                    → ``(1.5, 1.5)``
        * String ``"1.5 m"``                  → ``(1.5, 1.5)``
        * String ``"1.5-2.0"``                → ``(1.5, 2.0)``
        * String ``"1.5–2.0"`` (en-dash)      → ``(1.5, 2.0)``
        * Empty / NaN / unparseable           → ``(None, None)``

    The "continuation row" pattern (note text without a depth) is
    handled by the caller (``load_video_notes`` appends to the previous
    entry's note).
    """
    if token is None:
        return None, None
    if isinstance(token, (int, float)) and not pd.isna(token):
        v = float(token)
        return v, v
    if isinstance(token, str):
        s = token.strip()
        if not s:
            return None, None
        s_clean = re.sub(r"\s*m\s*$", "", s, flags=re.IGNORECASE)
        m = _RANGE_RE.match(s_clean)
        if m:
            return float(m.group("a")), float(m.group("b"))
        try:
            v = float(s_clean)
            return v, v
        except ValueError:
            return None, None
    return None, None
