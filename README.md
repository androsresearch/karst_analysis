uv# karst_analysis

Multi-method analysis of a coastal karst aquifer system at Andros, Bahamas.

This package supports a doctoral thesis investigating the freshwater–
saltwater interface and karst features (cavities, fractures) using
several complementary techniques. The central scientific contribution
is the **convergence** of these methods: identifying subsurface features
where multiple independent measurements agree.

Repository state: **v17.2**, 290 tests passing.

## Sub-packages

| Package | Status | Purpose |
|---|---|---|
| `karst_analysis.sec` | ★ Active | SEC profiles (YSI): loading, preprocessing (SavGol-segmented + LOWESS+PAVA), breakpoint detection, robustness analysis. |
| `karst_analysis.caliper` | ★ Active | Caliper logs: noise estimation, severity analysis, per-sample severity CSVs and master traces. |
| `karst_analysis.convergence` | ★ Active | Cross-technique panels: caliper × video, SEC × caliper × video, SEC × caliper by well (v11) and by site (v12), interactive HTML version (v14), quantitative SEC↔caliper matching. |
| `karst_analysis.corrections` | ★ Active | Datum transformations (vadose / GL / WT). |
| `karst_analysis.io` | ★ Active | Filename parsing + figure-output path resolver (v13 convention). |
| `karst_analysis.runs` | ★ Active | Run-tracking ledger. |
| `karst_analysis.videolog` | Stub | Borehole video logs. To be added. |
| `karst_analysis.ert` | ★ Active | ERT 1D resistivity profile loaders, breakpoint detection with deterministic seed discovery, mixing-zone identification (v16). SEC vs ERT comparison panels (v17). 2D inversion module still pending. |
| `karst_analysis.satellite` | Stub | Surface karst geomorphology. To be added. |
| `karst_analysis.drilling` | Stub | Drilling notes. To be added. |

## Installation

Requires Python ≥ 3.10.

```bash
# With uv (recommended)
uv sync --extra dev
uv pip install -e .
```

### Installing version patches (v16, v17, ...)

The project ships incremental changes as zip patches with a PowerShell
installer that backs up, copies, runs the test suite, and offers
rollback on failure. To install a patch (e.g. `v17.zip`):

```powershell
cd C:\Users\Mariana\Downloads
Expand-Archive .\v17.zip -DestinationPath .\v17 -Force
powershell -ExecutionPolicy Bypass -File .\v17\install_v17.ps1
```

Replace `v17` with the version you are installing. The installer
defaults the repo path to `C:\Users\Mariana\Documents\karst_analysis`;
override with `-RepoRoot <path>` if your repo lives elsewhere.

Backups are written to `<repo>\backups\v<N>_<timestamp>\` and are
retained even after a successful install, so you can audit or
manually roll back later.

## Quick start

```python
from karst_analysis.sec.io import load_ysi_csv
from karst_analysis.sec.preprocessing import process_savgol, process_lowess

# Note: from v14 onwards the raw CSVs live flat under data/raw/sec/<campaign>/
# (no /D subfolder).
df_raw = load_ysi_csv("data/raw/sec/2022_02/AW6_D_YSI_20220219.csv")
df_savgol, _ = process_savgol(df_raw)
df_lowess, _ = process_lowess(df_raw)
```

## Main flow — scripts (CLI)

The day-to-day workflow runs through batch scripts under `scripts/`.
All of them can be invoked with `uv run python scripts\<name>.py --help`
to see their options.

### Preprocessing

```powershell
# Run both smoothing pipelines (savgol, lowess) on a campaign's raw CSVs.
# Generates preprocessed CSVs + diagnostic figures.
uv run python scripts\preprocess_batch.py `
    --input data\raw\sec\2022_02 `
    --output data\processed\sec\2022_02
```

### Breakpoints

```powershell
# Compute breakpoints for all wells of a campaign (savgol + lowess, N=1..max).
uv run python scripts\breakpoints_batch.py --campaign 2022_02 --raw-dir data\raw\sec\2022_02

# Re-render breakpoint figures from existing .json files (no re-compute).
uv run python scripts\regenerate_breakpoint_figures.py --campaign 2022_02
```

### Robustness analysis

```powershell
# Cluster breakpoints between savgol and lowess; persistence and agreement metrics.
uv run python scripts\sec_robustness_analysis.py --campaign 2022_02
```

### Convergence panels

```powershell
# Caliper × video × Ardaman (pre-casing, no campaign):
uv run python scripts\caliper_video_panels.py

# SEC × caliper × video — preferred jobs-driven mode (v17.2):
#   - one panel per job in the YAML, honouring per-well trial/method/N
#   - BPs flagged TOP MZ / BOT MZ are coloured (red / purple) when the
#     matching slopes CSV exists; otherwise plain orange + warning
uv run python scripts\sec_caliper_video_panels.py --jobs config\slopes_jobs_2022_02.yml

# SEC × caliper × video — legacy grid mode (sensitivity sweep):
uv run python scripts\sec_caliper_video_panels.py --campaign 2022_02 --trial trial_1

# SEC × caliper panel by WELL, multi-campaign overlay (v11):
uv run python scripts\sec_caliper_panels.py --campaigns 2022_02

# SEC × caliper panel by SITE, multi-campaign overlay (v12):
uv run python scripts\site_panels.py

# Interactive HTML version of the SITE panel (v14):
uv run python scripts\site_panels_interactive.py

# SEC vs ERT 1D comparison panels per (well, transect, x) (v17):
uv run python scripts\sec_vs_ert_panels_batch.py --campaign 2022_02
```

### Quantitative SEC↔caliper matching

```powershell
# Per-cluster matching of robust SEC clusters against caliper severity zones.
uv run python scripts\sec_caliper_convergence.py --campaign 2022_02
```

### Diagnostics

Auxiliary scripts under `scripts/diagnostics/` to inspect intermediate
outputs without opening figures. Used during the trial-selection step
that feeds `config/slopes_jobs_<campaign>.yml`.

```powershell
# Tabulate BIC and breakpoint positions across the 3 trials per
# (well, method) at a given N. Useful to decide if the trials are
# consistent (any one is fine) or divergent (open the figures).
uv run python scripts\diagnostics\inspect_trials.py --campaign 2022_02 --n 15

# Render savgol-vs-lowess comparison figures for every (well, N, trial)
# combination of a campaign. Output: results/figures/breakpoints_trial_inspection/
# THROWAWAY: this script should be removed once the input-grid spatial
# balance problem is resolved (see NOTES_open_questions.md item 1) — at
# that point trial-to-trial spread should drop and per-trial inspection
# stops being meaningful.
uv run python scripts\diagnostics\render_all_trials.py --campaign 2022_02

# Generate three diagnostic figures for one well that evidence the
# spatial-balance problem: depth histogram of resampled grid points,
# density overlay on the log10(SEC) profile, and raw-vs-resample
# comparison. Currently hardcoded to LRS70D; use as a template by
# editing the well id at the top of the script.
uv run python scripts\diagnostics\quick_diagnose_lrs70d.py
```

## Notebooks (legacy, exploratory)

The repository ships with 10 numbered notebooks under `notebooks/`:

```
01_preprocess_batch.ipynb
02_compare_smoothing.ipynb
03_breakpoints_compute.ipynb
04_breakpoints_evaluate.ipynb
05_export_for_external_projects.ipynb
06_caliper_pipeline.ipynb
07_caliper_video_panels.ipynb
08_sec_caliper_video_panels.ipynb
09_sec_robustness_analysis.ipynb
10_sec_caliper_convergence.ipynb
```

**Status: NOT VERIFIED against any state of the repo from v15 onwards.**
They were the original interactive workflow before the codebase
consolidated around the `scripts/` CLIs in v10–v14. They are kept in
the repository because Mariana wants to preserve the option of
interactive data exploration, but they may need updating (paths
changed in v13, raw-data layout changed in v14, several new modules
and conventions were added in v15-v17) before they run cleanly.
Verifying / updating them is **not a priority** for the current
thesis push — it's a task for a future maintenance pass.

If you want to use one, expect to fix import paths and to re-point
hardcoded directories at the v13 figure-path convention.

## Stable API for cross-project consumption

External projects should import only from `karst_analysis.sec.export`:

```python
from karst_analysis.sec.export import (
    list_available_runs,      # what's on disk?
    load_sec_profile,         # smoothed trace, both datums
    load_breakpoints_at_n,    # BPs at any N, with CIs in both datums
    load_bic_curve,           # BIC vs N for the elbow decision
)
```

The functions take ``well_id``, ``campaign``, ``smoothing``, and an
optional ``n`` and ``project_root``. The ``n`` is chosen at call time —
you do **not** need to commit to one N before generating outputs.

## Run tracking

Every preprocessing or breakpoint run is recorded in `results/runs.csv`,
with a stable hash-based ``run_id`` and a human-readable method
signature embedded in output filenames. See `karst_analysis.runs`.

## Repository layout

```
src/karst_analysis/        # the package
notebooks/                 # legacy interactive workflows (see note above)
scripts/                   # batch CLIs — main flow today
data/                      # raw, processed, metadata, breakpoints
config/                    # pipeline_default.yml + optional pipeline.yml override
results/                   # all derived outputs
  ├─ runs.csv              #   project-wide run ledger
  ├─ figures/              #   ALL figure outputs (v13 convention)
  │    ├─ caliper/                              # pre-casing
  │    ├─ breakpoints/<campaign>/               # per-campaign
  │    ├─ diagnostic/<campaign>/                # per-campaign
  │    ├─ sec_robustness/<campaign>/            # per-campaign
  │    ├─ sensitivity_savgol_window/<campaign>/ # per-campaign
  │    └─ convergence/
  │         ├─ caliper_video/                   # pre-casing
  │         ├─ sec_caliper_video/<campaign>/    # per-campaign
  │         ├─ sec_caliper_panel/<sub>/         # <sub> = campaign or multi_Nc
  │         ├─ site_panel/<sub>/                # <sub> = campaign or multi_Nc
  │         ├─ site_panel_interactive/<sub>/    # HTML, v14
  │         └─ sec_ert/<campaign>/              # SEC vs ERT 1D panels, v17
  ├─ sec_robustness/<campaign>/                 # per-technique CSV outputs
  ├─ convergence/sec_caliper/<campaign>/        # quantitative SEC↔caliper match CSVs
  └─ sensitivity_savgol_window/<campaign>/      # CSVs of sensitivity sweep
legacy/                    # frozen pre-refactor code (do not edit)
tests/                     # 290 tests, all passing in v17.1
```

The figure-path convention introduced in v13 is
**`results/figures/<technique>/<sub>/`**:

* Pre-casing techniques (caliper, caliper_video) — no campaign
  subfolder. They were measured once per well before casing was
  installed and have no campaign concept.
* Single-campaign techniques (breakpoints, diagnostic, sec_robustness,
  sec_caliper_video, sensitivity_savgol_window) — one subfolder per
  campaign, named with the campaign id.
* Multi-campaign overlays (sec_caliper_panel, site_panel,
  site_panel_interactive) — one subfolder per campaign for
  single-campaign runs, or `multi_<N>c` when overlaying several
  campaigns in a single figure.

CSV outputs (non-figure) live in technique-specific paths under
`results/<technique>/<campaign>/` and were not moved by the v13
refactor.

## Tests

```powershell
uv run pytest tests\
```

Expected: **290 passed, 1 xfailed, 1 warning** in v17.1.
The 14 skipped tests in `test_ert_io.py` require the ERT fixtures
under `data/raw/ert/T16/1D/` to be present; they pass automatically
when the data is in place. The 1 xfailed test
(`test_tie_warning` in `test_ert_mixing_zone.py`) is intentional —
it documents a known issue inherited from SEC's mixing-zone tie
detection, tracked in `NOTES_open_questions.md` (entry #7).

The v17.2 changes (jobs-driven SEC × caliper × video panels with
mixing-zone–coloured breakpoints) are non-breaking for existing tests
— the legacy grid mode keeps working with all original defaults. See
`CHANGELOG.md` for the full description.

## License

Proprietary. © 2025 University of Bristol. See `LICENSE`.
