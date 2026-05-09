"""Geometric layout helpers for label placement on multi-row panels.

These functions are NOT specific to caliper or video-log data: they are
generic 1-D label-placement and bracket-drawing primitives. Keeping
them in their own module lets the v6+ panels (caliper + video + SEC)
reuse the same primitives without duplicating code.

Migration history
-----------------
v5.1: extracted from ``caliper_videolog_panel.py``. The PAV-isotonic
algorithm is preserved verbatim.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Minimum-displacement label positioning (PAV-based)
# ──────────────────────────────────────────────────────────────────────
def minimum_displacement_positions(
    anchors: np.ndarray,
    half_heights: np.ndarray,
    y_lo: float,
    y_hi: float,
    pad: float = 0.0,
) -> np.ndarray:
    """Choose label-centre y-positions that don't overlap each other.

    Each label has an *anchor* (the y-coordinate of the data point it
    is annotating) and a *half-height* (the vertical room it needs in
    plot units). The labels must not overlap. This routine returns the
    set of label centres that minimises the sum of squared
    displacements from the anchors, subject to:

        |y_i - y_j| >= h_i + h_j   for all i != j
        y_lo + h_i  <=  y_i  <=  y_hi - h_i

    The unconstrained minimum is the anchor itself. With the
    non-overlap constraints this is an isotonic regression problem in
    transformed coordinates, which the Pool-Adjacent-Violators (PAV)
    algorithm solves exactly in O(n).

    Parameters
    ----------
    anchors : np.ndarray
        Anchor y-coordinates (1-D).
    half_heights : np.ndarray
        Half-height of each label in plot units (1-D, same length).
    y_lo, y_hi : float
        Vertical bounds of the plotting area.
    pad : float, default 0.0
        Extra space between adjacent labels (added to each side, so
        total inter-label gap is ``pad``).

    Returns
    -------
    np.ndarray
        Label centre positions in the same order as ``anchors``.
    """
    n = len(anchors)
    if n == 0:
        return anchors.copy()

    # Sort by anchor descending (top of plot first).
    order = np.argsort(-anchors)
    a = anchors[order].astype(float)
    h = half_heights[order].astype(float) + 0.5 * pad

    # Cumulative spacing: S[k] is the minimum total height occupied by
    # labels 0..k-1 (their stacks touch).
    S = np.zeros(n)
    for k in range(1, n):
        S[k] = S[k - 1] + h[k - 1] + h[k]

    # Transformed coordinates: minimise sum (y_i - a_i)^2 with y_i
    # monotone non-increasing → isotonic on b = -a + S.
    b = -a + S

    # PAV: pool adjacent violators
    block_starts: list[int] = []
    block_vals:   list[float] = []
    block_wts:    list[float] = []
    for k in range(n):
        cur_val, cur_wt, cur_start = b[k], 1.0, k
        while block_vals and block_vals[-1] > cur_val:
            pv = block_vals.pop()
            pw = block_wts.pop()
            ps = block_starts.pop()
            new_wt = pw + cur_wt
            cur_val = (pv * pw + cur_val * cur_wt) / new_wt
            cur_wt = new_wt
            cur_start = ps
        block_vals.append(cur_val)
        block_wts.append(cur_wt)
        block_starts.append(cur_start)

    # Expand block solution back to per-element values
    v_out = np.empty(n)
    for j, (start, val, _) in enumerate(zip(block_starts, block_vals, block_wts)):
        end = block_starts[j + 1] if j + 1 < len(block_starts) else n
        v_out[start:end] = val
    y_sorted = -v_out + S

    # Top-down clamp + adjacency repair
    for k in range(n):
        if y_sorted[k] + h[k] > y_hi:
            y_sorted[k] = y_hi - h[k]
    for k in range(1, n):
        max_allowed = y_sorted[k - 1] - (h[k - 1] + h[k])
        if y_sorted[k] > max_allowed:
            y_sorted[k] = max_allowed
    for k in range(n - 1, -1, -1):
        if y_sorted[k] - h[k] < y_lo:
            y_sorted[k] = y_lo + h[k]
    for k in range(n - 2, -1, -1):
        min_allowed = y_sorted[k + 1] + (h[k + 1] + h[k])
        if y_sorted[k] < min_allowed:
            y_sorted[k] = min_allowed

    out = np.empty(n)
    out[order] = y_sorted
    return out


# ──────────────────────────────────────────────────────────────────────
#  Bracket drawing
# ──────────────────────────────────────────────────────────────────────
def draw_bracket(ax, x_anchor: float, x_tip: float,
                 y_top: float, y_bot: float, *, color: str, lw: float) -> None:
    """Draw a square bracket marking a y-interval.

    The bracket has a vertical spine at ``x_anchor`` and two short
    horizontal tips at the top and bottom (``y_top``, ``y_bot``), each
    extending from ``x_anchor`` to ``x_tip``. Used to indicate the
    extent of a depth interval (note or lithology) without ambiguity.
    """
    ax.plot([x_anchor, x_anchor], [y_top, y_bot], color=color, lw=lw,
            zorder=3, solid_capstyle="butt")
    ax.plot([x_anchor, x_tip], [y_top, y_top], color=color, lw=lw,
            zorder=3, solid_capstyle="butt")
    ax.plot([x_anchor, x_tip], [y_bot, y_bot], color=color, lw=lw,
            zorder=3, solid_capstyle="butt")


# ──────────────────────────────────────────────────────────────────────
#  Label-text builder
# ──────────────────────────────────────────────────────────────────────
def build_label_text(row) -> str:
    """Format a unified-entries row as a one-line label.

    Expects a row (``pd.Series``) with at least:
        depth_top_m, depth_bot_m, text, kind

    Special-cases the Ardaman kinds so the output starts with
    ``[Ardaman]`` for clarity in mixed panels.
    """
    z_top = row["depth_top_m"]
    z_bot = row["depth_bot_m"]
    if not np.isfinite(z_bot) or abs(z_bot - z_top) < 1e-6:
        depth_str = f"({z_top:.1f} m)"
    else:
        depth_str = f"({z_top:.1f}–{z_bot:.1f} m)"
    if row["kind"] in ("ardaman_lith", "ardaman_cond"):
        return f"[Ardaman] {depth_str} {row['text']}"
    return f"{depth_str} {row['text']}"
