# Changelog

All notable changes to `karst_analysis` are documented here. Older
versions (v1–v17.1) shipped as zip patches with PowerShell installers;
their internal notes live under `backups/`.

## v17.5 — 2026-05-31

### feat: SEC breakpoints summary CSV (cross-check table)

#### What changed

New script `scripts/build_sec_breakpoints_summary.py` produces a
long-format CSV with one row per breakpoint, listing both the
water-table depth (native frame of the JSONs and slopes CSVs) and
the BGL depth (= `depth_wt_m + vadose_m`), plus MZ flags from the
slopes CSV.

For each canonical (well, method, trial, N) job listed in
`config/slopes_jobs_<campaign>.yml`, the script:

1. Reads the breakpoints JSON to get the N water-table depths.
2. Reads the slopes CSV to get `is_top_of_mixing` /
   `is_bottom_of_mixing` flags.
3. Looks up vadose-zone thickness for the well from
   `data/metadata/wells.csv` (campaign-matched).
4. Cross-checks that JSON and slopes CSV describe the same series
   in WT datum (max disagreement < 1e-3 m).
5. Emits one row per BP with `well_id, campaign, date, method,
   trial, n_breakpoints, bp_idx, depth_wt_m, vadose_m,
   depth_bgl_m, is_top_mz, is_bot_mz`.

Output default: `results/sec_breakpoints_summary_<campaign>.csv`.
For campaign 2022_02 the snapshot is 75 rows (5 wells × 15 BPs).

#### Why

Two consumers needed a single cross-check table:

1. Thesis text. Specific BGL values are cited (LRS70D TOP MZ at
   9.11 m BGL, BOT MZ at 13.30 m BGL, etc.). Reading them from
   multiple JSONs + slopes CSVs is slow and error-prone; one CSV
   filtered by `is_top_mz=True` returns the TOP MZ of all wells
   in one row scan.
2. Companion fix in `andros_resipy_inversions` v0.1.0 (audit
   follow-up on BGL-vs-WT). Both repos now point at the same
   canonical numbers; the andros figures can be visually checked
   against the rows of this CSV.

The values are derived (script + inputs + commit), not new
computations. The breakpoints themselves were fitted in earlier
versions and remain canonical (Invariante 4).

#### What does NOT change

* No re-fit of breakpoints. The JSONs and slopes CSVs under
  `data/breakpoints/` and `data/slopes/` are read as-is.
* `data/metadata/wells.csv` is unchanged.
* No other scripts touched; pure addition.

#### Files added

* `scripts/build_sec_breakpoints_summary.py` — new.
* `results/sec_breakpoints_summary_2022_02.csv` — new (75 rows,
  campaign 2022_02 snapshot).
* `CHANGELOG.md` — this entry.

#### How to regenerate

```
uv run python scripts/build_sec_breakpoints_summary.py \
    --jobs config/slopes_jobs_2022_02.yml
```

---

## v17.4 — 2026-05-29

### feat: BIC curves panel — alphabetical layout + fixed N axis to expose non-convergence

#### What changed

* `karst_analysis.sec.robustness.viz.plot_bic_curves` gained an optional
  `n_max: int | None = None` keyword argument. When given, every
  subplot uses the same fixed x-axis range ``[0, n_max]`` with integer
  xticks at each step and ``xlim = (-0.5, n_max + 0.5)``. Subplots
  whose data does not extend to ``n_max`` show empty space on the
  right — this is deliberate, to visualise where a (well, smoothing)
  combination did not converge at higher N.
* When ``n_max`` is omitted (legacy path), the function still works
  but now uses a per-subplot dynamic upper bound (clamped at ≥ 10 for
  backwards compatibility with the v17.2 behaviour of hard-coded
  ``range(0, 11)`` xticks).
* `scripts/sec_robustness_analysis.py` now defaults its well order to
  ``sorted(WELLS.keys())`` instead of ``list(WELLS.keys())`` — so
  without ``--wells``, every per-well loop (BIC subplots, CSV rows,
  console output) processes wells alphabetically:
  AW5D → AW6D → BW3D → LRS69D → LRS70D. Passing ``--wells``
  explicitly preserves the user's order.
* The script now passes ``n_max=n_max`` (resolved from the config
  block ``robustness.n_max`` or the CLI flag ``--n-max``, which is
  ``15`` in the current ``config/pipeline.yml``) to
  ``plot_bic_curves``. So the rendered figure always uses the same
  N range that the rest of the robustness analysis uses.

#### Why

The previous BIC curves figure had two papercut bugs:

1. **xticks hardcoded to 0..10.** Even though the BIC-sweep JSONs
   already contain N up to 15 (the ``data/breakpoints/2022_02/*__bp-*-max15-t3.json``
   files), the figure only showed tick labels up to 10, so the curves
   appeared to extend past the labelled axis end with no way for the
   reader to read off where the BIC minimum landed for higher N.
2. **Subplot order followed dict declaration order**
   (LRS70D first because it was declared first in ``WELLS``), which
   is opaque to a reader. Alphabetical is the obvious default and
   matches how rows appear in the joined CSVs.

The fix also gives the figure a defensible scientific role: with a
fixed x-axis at N=0..15 across all 5 wells, an empty right segment in
any subplot directly exposes that the corresponding (well, smoothing)
fit did not converge to that N. That's a feature, not a flaw — the
thesis can now cite the BIC figure as evidence of where each pozo's
optimal complexity sits relative to the others.

#### Defaults / breaking change

* The function signature is back-compatible: callers that never
  passed ``n_max`` still work, just with the slightly improved legacy
  dynamic-with-floor-10 behaviour instead of the hard-coded 0..10.
* The script's CLI surface is unchanged. Default-invocation
  (``uv run python scripts/sec_robustness_analysis.py``) now produces
  the alphabetical, N=0..15 figure.

#### No re-computation of breakpoints

This patch is pure visualisation. It does NOT call
``breakpoints_batch.py`` and does NOT re-fit any BIC sweep. The
existing JSONs under ``data/breakpoints/2022_02/`` are read as-is.
This is important: the breakpoint detector is seed-sensitive
(documented in ``NOTES_open_questions.md``), so re-running it would
shift the BIC values by small but visible amounts. Plot-only fix
preserves the exact numbers that have already been cited downstream.

#### Files touched

* `src/karst_analysis/sec/robustness/viz.py` — modified.
* `scripts/sec_robustness_analysis.py` — modified.
* `CHANGELOG.md` — this entry.

---

## v17.3 — 2026-05-29

### fix: SEC figures now plot in below-ground-level datum (canonical)

#### What changed

Six visualisation functions gained a new `vadose_offset_m: float = 0.0`
keyword argument:

* `karst_analysis.sec.viz.slopes_overlay.plot_slopes_overlay`
* `karst_analysis.sec.viz.breakpoints_overlay.plot_breakpoints_overlay`
* `karst_analysis.sec.viz.breakpoints_overlay.plot_breakpoints_compare_methods`
* `karst_analysis.sec.viz.diagnostic.plot_diagnostic`
* `karst_analysis.sec.viz.diagnostic.plot_balance_histogram`
* `karst_analysis.sec.viz.comparison.plot_smoothing_comparison`

The parameter is added to the depth axis of every Y-coordinate the
function plots (raw scatter, smoothed line, breakpoint markers,
breakpoint label text, slopes_df `depth_top` / `depth_bottom`, zoom
ranges). The y-axis label is derived automatically from the offset:

* `vadose_offset_m > 0`  →  `"Depth below ground level (m)"`
* `vadose_offset_m == 0` (default)  →  `"Depth below water table (m)"`

For `plot_diagnostic`, `plot_balance_histogram`, and
`plot_smoothing_comparison`, the existing `depth_axis_label` argument
still wins if the caller passes a non-None value (escape hatch).

Five callers were updated to look up the well's
`vadose_thickness_m` from `data/metadata/wells.csv` via
`karst_analysis.corrections.get_vadose_thickness(well_id)` and pass it
through:

* `scripts/slopes_batch.py`
* `scripts/breakpoints_batch.py`
* `scripts/preprocess_batch.py`
* `scripts/regenerate_breakpoint_figures.py`
* `scripts/diagnostics/render_all_trials.py`

Each caller falls back to `vadose_offset_m = 0.0` (with a warning) when
the well isn't in `wells.csv`, so the scripts still produce a figure —
just in water-table datum with an honest label.

#### Why

Discovered while regenerating the SEC × caliper × video panel for
LRS70D (the thesis flagship case): the SEC-only branch of the figure
pipeline (slopes plots, breakpoint trial inspection, diagnostic
overlays) was plotting depth values straight from the SEC pipeline's
native `depth_m` column — which is referenced to the **water table**,
zero at the air-water interface inside the well — but labelling the
y-axis as `"Depth below ground level (m)"`. The convergence panels
(`sec_caliper_video`, `sec_caliper_panel`, `site_panel`,
`caliper/viz`) were already in BGL correctly, because they convert
via `data/metadata/wells.csv` before plotting.

This meant that for LRS70D, the breakpoint labelled "1.09 m below
ground level" in the slopes figure was actually at 1.92 m BGL
(1.09 m below the water table, where the water table sits 0.83 m
below ground level). The offset is the well's vadose-zone thickness.
For BW3D the offset is 3.28 m, large enough to materially misalign
breakpoints with the caliper anomalies and video-log features they
were supposedly converging onto.

BGL is the canonical datum for `karst_analysis` (now documented in
`working_style.md`) because every cross-technique anchor —
ground-level elevation, caliper trace zero, video-log observations,
Ardaman lithology, future ERT 2D — lives in BGL. Cross-technique
convergence requires a single datum and that datum has to be the one
the physical anchors use.

The SEC CSVs themselves (`data/processed/sec/`, `data/breakpoints/`,
`data/slopes/`) stay in water-table datum — they are the model's
native output and the source of scientific record. Only the figures
migrate.

#### Breaking change (label only, not data)

For any caller that did NOT pass `vadose_offset_m` after this patch,
the y-axis label changes from `"Depth below ground level (m)"`
(which was wrong) to `"Depth below water table (m)"` (which is
honest). The data plotted is unchanged in that case. To restore the
BGL label and have the depth values actually be BGL, the caller must
pass `vadose_offset_m=get_vadose_thickness(well_id)`. The five
scripts in this repo already do this.

#### Out of scope / deferred

* `notebooks/01_preprocess_batch.ipynb` and
  `notebooks/02_compare_smoothing.ipynb` call `plot_diagnostic` /
  `plot_smoothing_comparison` directly. They still work but produce
  figures labelled `"Depth below water table (m)"` until updated to
  pass `vadose_offset_m`. Tracked as follow-up; not blocking the
  thesis.
* The `breakpoints_trial_inspection` figures historically used in
  thesis-text references for LRS70D need re-generation; this is the
  first thing to run after applying the patch (see "Regeneration
  workflow" below).

#### Regeneration workflow for LRS70D

After applying this patch, regenerate every SEC figure that mentions
LRS70D so the thesis text references match the canonical datum:

```powershell
cd C:\Users\Mariana\Documents\karst_analysis
# 1. Slopes figures (uses the existing slopes_jobs YAML)
uv run python scripts\slopes_batch.py --jobs config\slopes_jobs_2022_02.yml
# 2. Trial-inspection figures for LRS70D
uv run python scripts\diagnostics\render_all_trials.py --campaign 2022_02 --only LRS70D
# 3. Convergence panels (already in BGL; safe to re-render for consistency)
uv run python scripts\sec_caliper_video_panels.py --jobs config\slopes_jobs_2022_02.yml
```

LRS70D BPs in BGL (LOWESS, N=15, trial 3) will be at: 1.92, 3.16,
4.07, 4.81, 9.11, 9.88, 11.69, 12.42, 12.78, 13.30, 14.10, 16.76,
22.09, 24.61, 26.40 m.

#### Files touched

* `src/karst_analysis/sec/viz/slopes_overlay.py` — modified.
* `src/karst_analysis/sec/viz/breakpoints_overlay.py` — modified.
* `src/karst_analysis/sec/viz/diagnostic.py` — modified.
* `src/karst_analysis/sec/viz/comparison.py` — modified.
* `scripts/slopes_batch.py` — passes `vadose_offset_m`.
* `scripts/breakpoints_batch.py` — passes `vadose_offset_m`.
* `scripts/preprocess_batch.py` — passes `vadose_offset_m`.
* `scripts/regenerate_breakpoint_figures.py` — passes `vadose_offset_m`.
* `scripts/diagnostics/render_all_trials.py` — passes `vadose_offset_m`.
* `working_style.md` — added §40 ("BGL es el datum canónico").
* `CHANGELOG.md` — this entry.

---

## v17.2 — 2026-05-29

### SEC × caliper × video panels: jobs-driven rendering + mixing-zone colouring

#### What changed

* `plot_sec_caliper_video_panel` (in
  `src/karst_analysis/convergence/sec_caliper_video.py`) gained a new
  `trial: str = "trial_1"` keyword argument. It is plumbed through to
  `load_breakpoints_at_n`, which already accepted `trial` but was
  hard-wired to the default by the panel.
* Breakpoint diamonds are now coloured per the mixing-zone flags
  produced by `scripts/slopes_batch.py`:
    * regular BP → orange (`#ff7f0e`)
    * TOP of mixing zone → deep red (`#c0392b`)
    * BOTTOM of mixing zone → purple (`#8e44ad`)
  The colours match `karst_analysis.sec.viz.slopes_overlay` exactly so
  the panel reads consistently with the standalone slopes figure
  generated by `slopes_batch.py`. Each BP-index label in column 1
  (`BP10: 12.5 m` etc.) inherits the marker colour, again mirroring
  `slopes_overlay`.
* The legend acquires `"TOP of mixing zone"` / `"BOTTOM of mixing zone"`
  entries only when the corresponding flag is active in the slopes
  CSV.
* The figure title now always includes the trial, rendered humanised:
  e.g. `(lowess, N=15, trial 3)`.
* `build_all_sec_caliper_video_panels` learnt a new operating mode
  (**jobs mode**) driven by the same YAML jobs file that already
  configures `scripts/slopes_batch.py`. Pass `jobs=[{well, method,
  trial, n}, ...]` and one panel is produced per job. The legacy grid
  mode (`wells × smoothings × [n_min..n_max]`) is preserved and gained
  a `trial` parameter (default `"trial_1"`) so a uniform trial can be
  swept across the grid.
* `scripts/sec_caliper_video_panels.py` learnt a `--jobs <yaml>` flag
  to drive jobs mode end-to-end. The legacy flags
  (`--wells / --smoothing / --n-min / --n-max`) plus a new `--trial`
  flag remain available; if both `--jobs` and any legacy flag are
  supplied, the legacy flags are ignored and a warning is printed.
* Output filenames now **always** carry a `__t{idx}` suffix:
  `{well}_{date}__{smoothing}__N{n:02d}__t{idx}.png`. This is a
  behavioural change for the legacy grid mode (previously
  `..._N{n:02d}.png`) — see "Breaking changes" below.

#### Why

The chapter protocol fixes one specific trial per well (e.g. LRS70D
uses `trial_3 N=15`, BW3D uses `trial_2 N=15`, etc., as captured in
`config/slopes_jobs_2022_02.yml`). Before v17.2 the convergence panel
silently used `trial_1` for every well, so the BP diamonds in the
three-technique panel disagreed with the BPs in the standalone slopes
figure for any well whose chapter trial was not 1. The panel also did
not visualise the TOP / BOT mixing-zone BPs that the slopes figure
already highlights — defeating one of the central messages of the
panel, namely that the mixing-zone boundaries identified from the SEC
profile coincide with caliper anomalies and video-log features.

Re-rendering the five priority panels now reduces to a single command:

```powershell
uv run python scripts\sec_caliper_video_panels.py --jobs config\slopes_jobs_2022_02.yml
```

#### Refactor

* A new module `src/karst_analysis/sec/jobs_io.py` owns the YAML jobs
  parser (`Job` dataclass, `load_jobs_file`, `trial_index`,
  `_normalise_campaign`). Previously this code lived inside
  `scripts/slopes_batch.py`. The script now imports from the shared
  module; behaviour is identical.
* The new module is the single source of truth for the
  `slopes_jobs_*.yml` schema and trial-to-index conversion. Both
  `slopes_batch.py` and `sec_caliper_video_panels.py` import from it.

#### Breaking changes

* **Output filename pattern for the SEC × caliper × video panel.**
  Old: `LRS70D_20220131__savgol__N10.png`.
  New: `LRS70D_20220131__savgol__N10__t1.png` (when trial = `trial_1`).
  Old PNGs are not deleted by the new code; they simply will not be
  overwritten on the next run. Manually delete the legacy-named files
  if you want a clean output directory.
* No public API was removed. `plot_sec_caliper_video_panel` and
  `build_all_sec_caliper_video_panels` accept all previous arguments
  with the same defaults; the new `trial` and `jobs` arguments are
  optional and default to backward-compatible values.

#### Graceful degradation

If the slopes CSV for a given (well, method, n, trial) combination is
missing under `data/slopes/<campaign>/`, the panel still renders, but:
* BPs are drawn as plain orange diamonds (no red / purple highlights).
* The legend has no TOP / BOT MZ entries.
* A `[warn]` line is printed pointing the user to run
  `scripts/slopes_batch.py` first.

This keeps the panel usable for ad-hoc sensitivity exploration where
the slopes step has not been run yet.

#### Files touched

* `src/karst_analysis/sec/jobs_io.py` — **new**.
* `src/karst_analysis/convergence/sec_caliper_video.py` — modified.
* `scripts/slopes_batch.py` — refactored to import from `jobs_io`;
  no behavioural change.
* `scripts/sec_caliper_video_panels.py` — added `--jobs` and
  `--trial` flags.
* `README.md` — updated the convergence-panels example.
* `CHANGELOG.md` — **new** (this file).
