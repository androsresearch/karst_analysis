"""
caliper_videolog_panel.py
=========================
Side-by-side panel: caliper raw signal + per-sample severity bands on the
left, video-log notes (and optional Ardaman lithology entries) on the
right. Built per priority well.

Inputs
------
* `priority_wells_cumulative_min_v2_perpoint.csv` — per-sample severities
* `Priority_Ewan_video_logs_v2.xlsx`              — cleaned video logs
* `ardaman_lithology.csv`                         — Ardaman 2009 transcripts

Conventions
-----------
* Caliper depths negative (0 = ground, downward = negative).
* Video-log and Ardaman depths positive in the source files; negated at
  load time to share the y-axis with the caliper.
* Video-logged well ≠ calipered well in general (D=deep, S=shallow,
  O=old/2009). Title makes the lateral correlation explicit.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ── Severity palette (matches priority_wells_cumulative_min_v2.py) ───────────
SEVERITY_COLORS = {"mild": "#fde3a7", "moderate": "#f39c12", "severe": "#c0392b"}
SEVERITY_ALPHAS = {"mild": 0.65, "moderate": 0.55, "severe": 0.55}
BAND_ZORDER = 1.5


# ── Per-well configuration ───────────────────────────────────────────────────
@dataclass
class WellConfig:
    caliper_well: str
    video_sheet:  str
    video_well:   str
    auger_cm:     float
    has_ardaman:  bool = False


WELLS: dict[str, WellConfig] = {
    "LRS70D": WellConfig("LRS70D", "LRS70D", "LRS70D", 20.32, has_ardaman=False),
    "AW5D":   WellConfig("AW5D",   "AW5",    "AW5O",   15.24, has_ardaman=True),
    "AW6D":   WellConfig("AW6D",   "AW6",    "AW6O",   15.24, has_ardaman=True),
    "BW3D":   WellConfig("BW3D",   "BW3S",   "BW3S",   15.24, has_ardaman=False),
    "LRS69D": WellConfig("LRS69D", "LRS69S", "LRS69S", 15.24, has_ardaman=False),
}


# ── Typo fixes for video-log text ────────────────────────────────────────────
TYPO_FIXES: list[tuple[str, str]] = [
    ("occuring",        "occurring"),
    ("beome",           "become"),
    ("Medium-lareg",    "Medium-large"),
    ("Eneter",          "Enter"),
    ("Botom",           "Bottom"),
    ("detriturs",       "detritus"),
    (", , ",            ", "),
    ("Meidum",          "Medium"),
    ("Smoo, ",          "Smooth, "),
    ("yellow.brown",    "yellow/brown"),
    ("beyonf",          "beyond"),
    ("salintiy",        "salinity"),
    ("largerdissolution", "larger dissolution"),
    ("Swiss chess",     "Swiss cheese"),
    ("intesifies",      "intensifies"),
    ("ocassional",      "occasional"),
    ("'sfc'",           "surface"),
    ("cloduy",          "cloudy"),
    ("Moertaely",       "Moderately"),
]


def _apply_typo_fixes(text: str) -> str:
    s = text
    for pat, rep in TYPO_FIXES:
        s = s.replace(pat, rep)
    return s.strip()


# ── Depth parsing ────────────────────────────────────────────────────────────
_RANGE_RE = re.compile(
    r"^\s*(?P<a>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<b>\d+(?:\.\d+)?)\s*$")


def _parse_depth_token(token) -> tuple[float | None, float | None]:
    """Return (z_top, z_bot) in positive metres. Single point ⇒ both equal."""
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


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_perpoint(perpoint_csv, well: str) -> pd.DataFrame:
    df = pd.read_csv(perpoint_csv)
    sub = df[df["well"] == well].copy()
    if sub.empty:
        raise ValueError(f"No per-sample rows for '{well}'")
    return sub.sort_values("depth_m", ascending=False).reset_index(drop=True)


# ── Companion-caliper styling ────────────────────────────────────────────────
# The deep well (suffix D) is the one with severity bands and the brown
# trace. The other companions at the same site (suffix O and S) are
# overlaid as additional traces in plain colours so the user can compare
# the raw geometry of paired boreholes that are not the calipered one.
COMPANION_STYLE: dict[str, dict] = {
    "O": dict(color="#000000", lw=0.6, alpha=0.85),  # old well (2009 era), black
    "S": dict(color="#444444", lw=0.6, alpha=0.85),  # shallow companion, dark grey
}


def _site_prefix(well_name: str) -> str:
    """
    Return the site prefix shared by D/O/S companions.

    Examples
    --------
    >>> _site_prefix("AW5D")
    'AW5'
    >>> _site_prefix("LRS69D")
    'LRS69'
    """
    m = re.match(r"^([A-Z]+\d+)", well_name)
    if m is None:
        raise ValueError(f"Cannot parse site prefix from '{well_name}'")
    return m.group(1)


def load_companions_caliper(master_csv: str | Path,
                            primary_well: str) -> dict[str, pd.DataFrame]:
    """
    Read the master concatenated caliper CSV and return all caliper
    traces for the same site as ``primary_well``, EXCLUDING the
    primary itself. Returns a dict ``{companion_well: DataFrame}``
    where the DataFrame has columns ``depth_m`` (negative metres,
    elevation) and ``caliper_cm``. The site is identified by
    stripping the trailing D/O/S suffix from ``primary_well``.

    If multiple LAS files exist for the same companion well (e.g.
    ``BW10D_caliper_20210922.LAS`` and ``BW10D_caliper_20211020.LAS``),
    they are concatenated and sorted by depth. The most common case
    has one LAS per companion.
    """
    df = pd.read_csv(master_csv)
    df["well"] = df["source_file"].str.split("_").str[0]
    site = _site_prefix(primary_well)
    # Match companions whose name starts with the site prefix and is
    # immediately followed by a single suffix letter (not another digit).
    pattern = re.compile(rf"^{re.escape(site)}[A-Z]$")
    out: dict[str, pd.DataFrame] = {}
    for w in sorted(df["well"].unique()):
        if w == primary_well:
            continue
        if not pattern.match(w):
            continue
        sub = (df[df["well"] == w][["Depth [m]", "calibrated_cm"]]
               .rename(columns={"Depth [m]": "depth_m",
                                "calibrated_cm": "caliper_cm"})
               .sort_values("depth_m")
               .reset_index(drop=True))
        if sub.empty:
            continue
        out[w] = sub
    return out


def caliper_from_perpoint(perpoint_df, auger_cm: float, iqr_k: float = 1.5):
    df = pd.DataFrame({
        "depth_m":        perpoint_df["depth_m"].to_numpy(),
        "raw_caliper_cm": perpoint_df["caliper_cm"].to_numpy(),
    }).sort_values("depth_m").reset_index(drop=True)
    cal = df["raw_caliper_cm"].to_numpy()
    valid = cal >= auger_cm
    if valid.any():
        q1 = float(np.nanpercentile(cal[valid], 25))
        q3 = float(np.nanpercentile(cal[valid], 75))
        lower_fence = q1 - iqr_k * (q3 - q1)
    else:
        lower_fence = -np.inf
    df["is_outlier"] = (cal < auger_cm) & (cal < lower_fence)
    return df


def load_video_notes(xlsx_path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    depth_col = next((c for c in raw.columns if "Depth" in c), None)
    notes_col = next((c for c in raw.columns
                      if c.lower().startswith("notes")), None)
    if depth_col is None or notes_col is None:
        raise KeyError(f"Sheet '{sheet}' missing Depth/Notes columns")

    rows: list[dict] = []
    for _, r in raw.iterrows():
        depth_token = r[depth_col]
        note = r[notes_col]
        if pd.isna(note) or not str(note).strip():
            continue
        z_top, z_bot = _parse_depth_token(depth_token)
        if z_top is None:
            # Continuation row → append to previous entry's note (option A)
            if rows:
                rows[-1]["note"] = (rows[-1]["note"].rstrip(". ").rstrip()
                                    + "; " + str(note).strip())
            continue
        rows.append(dict(depth_top_m=z_top, depth_bot_m=z_bot,
                         note=str(note).strip()))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["note"] = df["note"].map(_apply_typo_fixes)
    df["elev_top_m"] = -df["depth_top_m"]
    df["elev_bot_m"] = -df["depth_bot_m"]
    df["elev_centre_m"] = 0.5 * (df["elev_top_m"] + df["elev_bot_m"])
    return df.sort_values("elev_centre_m", ascending=False).reset_index(drop=True)


def load_ardaman(csv_path, well: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", skip_blank_lines=True)
    sub = df[df["well"] == well].sort_values("depth_m").reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()
    is_lith = sub["kind"] == "lithology"
    lith_idx = sub.index[is_lith].tolist()
    bot = sub["depth_m"].to_numpy(dtype=float).copy()
    for k, idx in enumerate(lith_idx):
        if k + 1 < len(lith_idx):
            bot[idx] = float(sub.loc[lith_idx[k + 1], "depth_m"])
        else:
            bot[idx] = np.nan
    for idx in sub.index[~is_lith]:
        bot[idx] = float(sub.loc[idx, "depth_m"])
    out = pd.DataFrame({
        "depth_top_m": sub["depth_m"].to_numpy(dtype=float),
        "depth_bot_m": bot,
        "kind":        sub["kind"].to_numpy(dtype=object),
        "text":        sub["text"].to_numpy(dtype=object),
    })
    out["elev_top_m"] = -out["depth_top_m"]
    out["elev_bot_m"] = -out["depth_bot_m"]
    out["elev_centre_m"] = np.where(
        np.isfinite(out["elev_bot_m"]),
        0.5 * (out["elev_top_m"] + out["elev_bot_m"]),
        out["elev_top_m"],
    )
    return out.sort_values("elev_centre_m", ascending=False).reset_index(drop=True)


# ── Severity-band run-length encoding ────────────────────────────────────────
def _severity_runs(depths, severities, levels=("mild", "moderate", "severe")):
    n = len(depths)
    if n == 0:
        return []
    half_dz = (0.5 * float(np.median(np.abs(np.diff(np.sort(depths)))))
               if n > 1 else 0.015)
    runs: list[tuple[str, float, float]] = []
    i = 0
    while i < n:
        sev = severities[i]
        if sev not in levels:
            i += 1
            continue
        j = i
        while j + 1 < n and severities[j + 1] == sev:
            j += 1
        runs.append((sev, depths[i] + half_dz, depths[j] - half_dz))
        i = j + 1
    return runs


# ── PAV-based minimum-displacement label placement ───────────────────────────
def _minimum_displacement_positions(anchors, half_heights, y_lo, y_hi, pad=0.0):
    n = len(anchors)
    if n == 0:
        return anchors.copy()
    order = np.argsort(-anchors)
    a = anchors[order].astype(float)
    h = half_heights[order].astype(float) + 0.5 * pad
    S = np.zeros(n)
    for k in range(1, n):
        S[k] = S[k - 1] + h[k - 1] + h[k]
    b = -a + S

    block_starts: list[int] = []
    block_vals:   list[float] = []
    block_wts:    list[float] = []
    for k in range(n):
        cur_val, cur_wt, cur_start = b[k], 1.0, k
        while block_vals and block_vals[-1] > cur_val:
            pv = block_vals.pop(); pw = block_wts.pop(); ps = block_starts.pop()
            new_wt = pw + cur_wt
            cur_val = (pv * pw + cur_val * cur_wt) / new_wt
            cur_wt = new_wt
            cur_start = ps
        block_vals.append(cur_val)
        block_wts.append(cur_wt)
        block_starts.append(cur_start)

    v_out = np.empty(n)
    for j, (start, val, _) in enumerate(zip(block_starts, block_vals, block_wts)):
        end = block_starts[j + 1] if j + 1 < len(block_starts) else n
        v_out[start:end] = val
    y_sorted = -v_out + S

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


# ── Plot config ──────────────────────────────────────────────────────────────
@dataclass
class PanelConfig:
    base_figsize: tuple = (13, 14)              # baseline figure size
    width_ratio: tuple = (1.0, 3.2)             # caliper : notes — more text space
    note_wrap: int = 110                        # chars per line
    note_fontsize: float = 8.5
    note_x: float = 0.05                        # x where note text starts
    leader_color: str = "#888888"
    leader_lw: float = 0.55
    bracket_lw: float = 1.3
    bracket_tip: float = 0.008
    outlier_color: str = "#7a1ea8"
    grid_alpha: float = 0.22
    note_color: str = "#1a1a1a"
    ardaman_color_lith: str = "#1d4ed8"
    ardaman_color_cond: str = "#0f7a4d"
    # Auto-height tuning. Required height per label (data-units * inches/data)
    # is roughly fontsize_pt * 1.3 / 72 inches per single-line label. With
    # ~2 lines average we need ~0.3 in per label. Add ~50% safety. Floor at
    # `base_figsize[1]`, no upper bound.
    auto_height_factor: float = 0.30            # in/label
    auto_height_safety:  float = 1.50


def _draw_severity_bands(ax, perpoint_df, *, alpha_factor=1.0):
    runs = _severity_runs(
        perpoint_df["depth_m"].to_numpy(),
        perpoint_df["severity_per_sample"].to_numpy(dtype=object),
    )
    draw_order = {"mild": 0, "moderate": 1, "severe": 2}
    for sev, z_top, z_bot in sorted(runs, key=lambda r: draw_order.get(r[0], 0)):
        ax.axhspan(z_bot, z_top,
                   color=SEVERITY_COLORS.get(sev, "#cccccc"),
                   alpha=SEVERITY_ALPHAS.get(sev, 0.35) * alpha_factor,
                   zorder=BAND_ZORDER, linewidth=0)


def _draw_bracket(ax, x_anchor, x_tip, e_top, e_bot, *, color, lw):
    ax.plot([x_anchor, x_anchor], [e_top, e_bot], color=color, lw=lw, zorder=3,
            solid_capstyle="butt")
    ax.plot([x_anchor, x_tip], [e_top, e_top], color=color, lw=lw, zorder=3,
            solid_capstyle="butt")
    ax.plot([x_anchor, x_tip], [e_bot, e_bot], color=color, lw=lw, zorder=3,
            solid_capstyle="butt")


def _build_label_text(row: pd.Series) -> str:
    z_top = row["depth_top_m"]
    z_bot = row["depth_bot_m"]
    if not np.isfinite(z_bot) or abs(z_bot - z_top) < 1e-6:
        depth_str = f"({z_top:.1f} m)"
    else:
        depth_str = f"({z_top:.1f}–{z_bot:.1f} m)"
    if row["kind"] in ("ardaman_lith", "ardaman_cond"):
        return f"[Ardaman] {depth_str} {row['text']}"
    return f"{depth_str} {row['text']}"


# ── Main plot function ───────────────────────────────────────────────────────
def plot_panel(well_id: str, perpoint_csv, video_xlsx,
               ardaman_csv=None, master_caliper_csv=None,
               config=None, out_path=None,
               sat_cm: float = 32.5) -> plt.Figure:
    cfg = config or PanelConfig()
    if well_id not in WELLS:
        raise KeyError(f"Unknown well '{well_id}'")
    wc = WELLS[well_id]

    perpoint_df = load_perpoint(perpoint_csv, well=wc.caliper_well)
    cal_df = caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)
    notes_df = load_video_notes(video_xlsx, sheet=wc.video_sheet)
    ardaman_df = (load_ardaman(ardaman_csv, well=wc.video_well)
                  if (wc.has_ardaman and ardaman_csv is not None)
                  else pd.DataFrame())

    # Companion caliper traces from the master concatenated file
    # (same site, different suffixes — typically D's siblings are O and S).
    companions = (load_companions_caliper(master_caliper_csv,
                                          primary_well=wc.caliper_well)
                  if master_caliper_csv is not None else {})

    # Unified entries DataFrame
    parts = []
    if not notes_df.empty:
        nd = notes_df.copy()
        nd["kind"] = "note"
        parts.append(nd[["depth_top_m", "depth_bot_m", "elev_top_m",
                         "elev_bot_m", "elev_centre_m", "kind", "note"]]
                     .rename(columns={"note": "text"}))
    if not ardaman_df.empty:
        ad = ardaman_df.copy()
        ad["kind"] = ad["kind"].map({
            "lithology": "ardaman_lith",
            "conductivity_in_situ": "ardaman_cond"})
        parts.append(ad[["depth_top_m", "depth_bot_m", "elev_top_m",
                         "elev_bot_m", "elev_centre_m", "kind", "text"]])
    entries = (pd.concat(parts, ignore_index=True)
               .sort_values("elev_centre_m", ascending=False)
               .reset_index(drop=True)) if parts else pd.DataFrame()

    # Compute dynamic figure height based on total entries to place
    n_entries = len(entries) if not entries.empty else 0
    base_w, base_h = cfg.base_figsize
    needed_h = cfg.auto_height_factor * cfg.auto_height_safety * n_entries
    fig_h = max(base_h, needed_h)
    figsize = (base_w, fig_h)

    fig, (ax_cal, ax_note) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratio, wspace=0.02))

    # Y-limits from union of all sources
    ymin_c = [cal_df["depth_m"].min()]
    ymax_c = [cal_df["depth_m"].max()]
    if not entries.empty:
        ymin_c.append(float(entries["elev_centre_m"].min()))
        if entries["elev_bot_m"].notna().any():
            ymin_c.append(float(np.nanmin(entries["elev_bot_m"])))
        ymax_c.append(float(entries["elev_centre_m"].max()))
        if entries["elev_top_m"].notna().any():
            ymax_c.append(float(np.nanmax(entries["elev_top_m"])))
    y_min = min(ymin_c) - 0.8
    y_max = max(ymax_c) + 0.8
    ax_cal.set_ylim(y_min, y_max)

    # LEFT panel
    _draw_severity_bands(ax_cal, perpoint_df, alpha_factor=1.0)

    # Companion traces (drawn FIRST, so the primary D trace and the
    # severity bands stay visually on top). One line per companion;
    # styling looked up by suffix in COMPANION_STYLE.
    companion_handles = []
    for cmp_name, cmp_df in companions.items():
        suffix = cmp_name[len(_site_prefix(cmp_name)):]
        style = COMPANION_STYLE.get(suffix)
        if style is None:
            continue   # unknown suffix — skip rather than guess
        ax_cal.plot(cmp_df["caliper_cm"].to_numpy(),
                    cmp_df["depth_m"].to_numpy(),
                    label=cmp_name, zorder=3, **style)
        companion_handles.append((cmp_name, style))

    # Primary D trace (on top of companions)
    z = cal_df["depth_m"].to_numpy()
    raw = cal_df["raw_caliper_cm"].to_numpy()
    out = cal_df["is_outlier"].to_numpy().astype(bool)
    cal_plot = np.where(out, np.nan, raw)
    ax_cal.plot(cal_plot, z, color="#8e6914", lw=0.6, alpha=0.85,
                zorder=4, label=wc.caliper_well)
    if out.any():
        ax_cal.scatter(raw[out], z[out], s=40, marker="x",
                       c=cfg.outlier_color, zorder=5, linewidths=1.0)
    ax_cal.axvline(sat_cm, color="#777777", lw=0.6, ls=":", alpha=0.55, zorder=2)
    ax_cal.set_xlabel("Caliper aperture (cm)", fontsize=10)
    ax_cal.set_ylabel("Elevation (m)", fontsize=10)
    ax_cal.grid(True, axis="both", alpha=cfg.grid_alpha, linestyle=":")
    ax_cal.tick_params(axis="both", labelsize=8)
    ax_cal.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # X-range: must accommodate ALL traces, not just the primary.
    # Companions perforated with smaller augers (e.g. AW5O/AW6O are 4")
    # have caliper values down to ~10 cm, so we widen x_lo when needed.
    # Outliers from the primary's caliper are excluded from the lower
    # bound (they're marked with x but shouldn't drag the axis).
    all_caliper_mins = [np.nanmin(raw[~out]) if (~out).any() else np.nanmax(raw)]
    all_caliper_maxs = [np.nanmax(raw)]
    for cmp_df in companions.values():
        c = cmp_df["caliper_cm"].to_numpy()
        if c.size:
            all_caliper_mins.append(float(np.nanmin(c)))
            all_caliper_maxs.append(float(np.nanmax(c)))
    x_lo = min(min(all_caliper_mins) - 1.0,
               wc.auger_cm - 1.0)
    x_hi = max(max(all_caliper_maxs), sat_cm) + 1.5
    ax_cal.set_xlim(x_lo, x_hi)

    handles = []
    # Primary D trace handle
    handles.append(Line2D([0], [0], color="#8e6914", lw=1.2,
                          label=wc.caliper_well))
    # Companion handles (in alphabetical order: D, O, S → D was first;
    # so companions appear after the primary)
    for cmp_name, style in companion_handles:
        handles.append(Line2D([0], [0],
                              color=style["color"],
                              lw=max(style["lw"], 1.0),
                              alpha=style.get("alpha", 1.0),
                              label=cmp_name))
    # Severity patches (only those actually present)
    for sev in ("mild", "moderate", "severe"):
        if (perpoint_df["severity_per_sample"] == sev).any():
            handles.append(Patch(facecolor=SEVERITY_COLORS[sev],
                                 alpha=SEVERITY_ALPHAS[sev],
                                 edgecolor="none",
                                 label=sev.capitalize()))
    if handles:
        ax_cal.legend(handles=handles, loc="upper left", fontsize=7.5,
                      framealpha=0.92, edgecolor="#cccccc",
                      handlelength=1.5, handleheight=1.0,
                      borderpad=0.4, bbox_to_anchor=(0.0, 1.0))

    # RIGHT panel
    ax_note.set_xlim(0, 1)
    ax_note.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_note.tick_params(axis="y", which="both", left=False, labelleft=False)
    for spine in ("top", "right", "bottom"):
        ax_note.spines[spine].set_visible(False)
    ax_note.spines["left"].set_color("#bbbbbb")
    _draw_severity_bands(ax_note, perpoint_df, alpha_factor=0.45)

    if not entries.empty:
        labels = [_build_label_text(row) for _, row in entries.iterrows()]
        colors = entries["kind"].map({
            "note":         cfg.note_color,
            "ardaman_lith": cfg.ardaman_color_lith,
            "ardaman_cond": cfg.ardaman_color_cond,
        }).tolist()
        styles = entries["kind"].map({
            "note":         "normal",
            "ardaman_lith": "italic",
            "ardaman_cond": "italic",
        }).tolist()

        wrapped = [textwrap.fill(s, width=cfg.note_wrap) for s in labels]
        line_counts = np.array([w.count("\n") + 1 for w in wrapped])

        y_span = y_max - y_min
        axis_h_in = figsize[1] * 0.90
        pt_per_data = 72 * axis_h_in / y_span
        line_h_data = (cfg.note_fontsize * 1.30) / pt_per_data
        label_heights = line_counts * line_h_data + 0.7 * line_h_data
        half_heights = 0.5 * label_heights

        anchors = entries["elev_centre_m"].to_numpy()
        text_y = _minimum_displacement_positions(
            anchors, half_heights,
            y_lo=y_min + 0.4, y_hi=y_max - 0.4,
            pad=0.04 * line_h_data,
        )

        x_anchor = 0.005
        x_kink   = 0.025
        x_text   = cfg.note_x
        text_x_left = x_text + 0.005

        for i, (_, row) in enumerate(entries.iterrows()):
            e_top = row["elev_top_m"]
            e_bot = row["elev_bot_m"]
            anchor_y = row["elev_centre_m"]
            color = colors[i]
            fontstyle = styles[i]
            wrapped_text = wrapped[i]
            ty = text_y[i]

            is_interval = (np.isfinite(e_bot)
                           and abs(e_top - e_bot) > 1e-6)
            if is_interval:
                _draw_bracket(ax_note, x_anchor, x_anchor + cfg.bracket_tip,
                              e_top, e_bot, color=color, lw=cfg.bracket_lw)
                leader_x_start = x_anchor + cfg.bracket_tip
            else:
                ax_note.plot(x_anchor, anchor_y, marker="o", ms=2.8,
                             color=color, zorder=3)
                leader_x_start = x_anchor

            ax_note.plot([leader_x_start, x_kink, x_text],
                         [anchor_y, anchor_y, ty],
                         color=cfg.leader_color, lw=cfg.leader_lw,
                         alpha=0.85, zorder=2)
            ax_note.text(text_x_left, ty, wrapped_text,
                         ha="left", va="center",
                         fontsize=cfg.note_fontsize,
                         color=color, fontstyle=fontstyle, zorder=4,
                         bbox=dict(boxstyle="round,pad=0.28",
                                   facecolor="white", edgecolor="#dddddd",
                                   alpha=0.88, linewidth=0.5))

    right_label = "Video-log observations"
    if wc.has_ardaman:
        right_label += " (black) + Ardaman 2009 (blue / green)"
    ax_note.set_xlabel(right_label, fontsize=10)

    if wc.video_well == wc.caliper_well:
        title = (f"Well {wc.caliper_well} — caliper breakout zones "
                 f"and video-log observations")
    else:
        ard_note = (f"; Ardaman 2009 ({wc.video_well})" if wc.has_ardaman else "")
        title = (f"Well {wc.caliper_well} caliper ")   
                
    # Place suptitle ~0.4 inch from top of figure
    title_y = 1.0 - 0.4 / figsize[1]
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=title_y)
    # Title margin: reserve a fixed ~1.0 inch at the top for the suptitle
    # regardless of figure height. Makes the title stay fully visible
    # when the figure grows tall for high-density wells.
    top_margin = max(0.92, 1.0 - 1.0 / figsize[1])
    fig.subplots_adjust(left=0.075, right=0.985, top=top_margin, bottom=0.05)

    if out_path is not None:
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        print(f"  saved: {out_path}")
    return fig


def build_all(data_dir: str | Path = ".",
              out_dir:  str | Path = "./outputs",
              wells:    list[str] | None = None) -> None:
    """
    Build all priority-well panels.

    Parameters
    ----------
    data_dir : directory containing the three input files. Defaults to
        the current working directory.
    out_dir  : directory where PNGs will be written. Created if absent.
    wells    : optional subset of wells to render. Defaults to all in WELLS.

    Required input files in ``data_dir``:
        - priority_wells_cumulative_min_v2_perpoint.csv
        - Priority_Ewan_video_logs_v2.xlsx
        - ardaman_lithology.csv
        - concatenate_caliper_all.csv  (master caliper for companion traces)
    """
    data_dir = Path(r".\notebooks\sandbox\08_caliper_video_ODS")
    out_dir  = Path(r".\notebooks\sandbox\08_caliper_video_ODS\outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    perpoint = data_dir / "priority_wells_cumulative_min_v2_perpoint.csv"
    video    = data_dir / "Priority_Ewan_video_logs_v2.xlsx"
    ardaman  = data_dir / "ardaman_lithology.csv"
    master   = Path(r"data\caliper\concatenate_caliper_all.csv")

    for f, name in [(perpoint, "perpoint"), (video, "video"),
                    (ardaman, "ardaman"), (master, "master caliper")]:
        if not f.exists():
            raise FileNotFoundError(f"Missing {name} file: {f}")

    target_wells = wells if wells is not None else list(WELLS.keys())
    for w in target_wells:
        if w not in WELLS:
            print(f"\n[{w}]  unknown — skipping")
            continue
        print(f"\n[{w}]")
        try:
            fig = plot_panel(
                well_id=w,
                perpoint_csv=perpoint,
                video_xlsx=video,
                ardaman_csv=ardaman if WELLS[w].has_ardaman else None,
                master_caliper_csv=master,
                out_path=out_dir / f"{w}_caliper_videolog_panel.png",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build caliper + video-log + Ardaman panels for "
                    "the priority wells.")
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing the input CSV/XLSX files "
             "(default: current directory).")
    parser.add_argument(
        "--out-dir", default="./outputs",
        help="Output directory for the PNG panels "
             "(default: ./outputs).")
    parser.add_argument(
        "--wells", nargs="+", default=None,
        choices=list(WELLS.keys()),
        help="Subset of wells to render (default: all).")
    args = parser.parse_args()
    build_all(data_dir=args.data_dir, out_dir=args.out_dir, wells=args.wells)
