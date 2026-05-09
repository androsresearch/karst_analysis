"""SavGol window sensitivity scan.

Runs the FULL pipeline (preprocess → breakpoints → robustness) for
several values of the SavGol window length, and produces a comparative
figure showing how the top robust breakpoints shift (or don't) as the
smoothing window changes.

Why this matters
----------------
The SavGol window is the most consequential preprocessing choice for
breakpoint detection: a too-small window leaves ringing artefacts that
spawn false breakpoints; a too-large window smears genuine transitions
and hides true breakpoints. A defensible thesis claim about a specific
breakpoint depth must show that the breakpoint survives reasonable
variation of this parameter.

Defaults
--------
windows = [7, 11, 15, 21, 31]
For each window, the script writes outputs under separate folders
(processed CSVs, breakpoints, robustness) so nothing is overwritten.

Usage
-----
    uv run python scripts/sensitivity_savgol_window.py \\
        --raw-dir data/raw/sec/2022_02 \\
        --campaign 2022_02

    # Or a custom subset of windows
    uv run python scripts/sensitivity_savgol_window.py \\
        --raw-dir data/raw/sec/2022_02 \\
        --campaign 2022_02 \\
        --windows 11 21
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from karst_analysis.config import default_config_path, load_config
from karst_analysis.convergence import WELLS


SENSITIVITY_DIR_NAME = "savgol_window_sensitivity"


def _run(cmd: list[str]) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def _generate_config(window: int, out_dir: Path) -> Path:
    """Write a minimal YAML override config for the given window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"window{window}.yml"
    p.write_text(
        f"# Sensitivity test: SavGol window = {window}\n"
        f"# All other parameters fall back to pipeline_default.yml.\n"
        f"preprocessing:\n"
        f"  savgol:\n"
        f"    window: {window}\n"
    )
    return p


def _run_pipeline_for_window(
    window: int,
    *,
    raw_dir: Path,
    campaign: str,
    base_out: Path,
    config_dir: Path,
) -> Path:
    """Run preprocess → breakpoints → robustness for a single window.

    Returns the robustness output dir (where robustness_clusters.csv lives).
    """
    cfg_path = _generate_config(window, config_dir)

    # Outputs use a window-specific suffix so they coexist with the
    # default-window run (and with each other).
    proc_root = base_out / "processed_sec" / f"window{window:02d}"
    bp_dir = base_out / "breakpoints" / f"window{window:02d}"
    rob_dir = base_out / "robustness" / f"window{window:02d}"

    rc = _run([
        sys.executable, "scripts/preprocess_batch.py",
        "--input", str(raw_dir),
        "--output", str(proc_root),
        "--config", str(cfg_path),
    ])
    if rc != 0:
        raise RuntimeError(f"preprocess failed for window={window} (rc={rc})")

    rc = _run([
        sys.executable, "scripts/breakpoints_batch.py",
        "--raw-dir", str(raw_dir),
        "--config", str(cfg_path),
        "--proc-root", str(proc_root),
        "--bp-dir", str(bp_dir),
    ])
    if rc != 0:
        raise RuntimeError(f"breakpoints failed for window={window} (rc={rc})")

    # Robustness reads SEC + breakpoints from project-relative paths
    # (see sec.export.api). To make it use the per-window outputs, we
    # construct a temporary project layout via env-passing — but it's
    # easier to just point the robustness output dir per-window and let
    # it read the per-window data via project_root override. The
    # sec_robustness_analysis script doesn't expose that, so instead we
    # symlink the per-window data into data/processed/sec/<campaign>/
    # for the run, then symlink back. That's brittle.
    #
    # Cleaner path: use the existing CLI but tell it where the data is
    # via --output-dir; the robustness module reads from the standard
    # data/processed/sec/<campaign>/ and data/breakpoints/<campaign>/
    # paths. So we must temporarily make those paths the per-window ones.
    # The simplest way: symlinks (Linux) or copy (Windows). For safety
    # and cross-platform robustness here, we COPY the per-window data
    # into the standard paths in a "swap" pattern. That's intrusive.
    #
    # Instead, we expose project_root via the LOAD path. compute_robustness
    # accepts project_root. We'll call it directly here, not via the CLI.

    # ── Direct call into the API for robustness ─────────────────────
    from karst_analysis.sec.robustness import (
        compute_robustness, compute_robustness_sensitivity,
        plot_robustness_panel, plot_delta_sensitivity,
    )

    # We need a "project root" where data/processed/sec/<campaign>/<smoothing>/
    # and data/breakpoints/<campaign>/ point to the per-window data.
    # Build a staging dir with the right layout via symlinks.
    stage = base_out / f"_stage_window{window:02d}"
    if stage.exists():
        # Clean it
        import shutil
        shutil.rmtree(stage)
    stage_data_proc = stage / "data" / "processed" / "sec" / campaign
    stage_data_bp = stage / "data" / "breakpoints" / campaign
    stage_data_proc.parent.mkdir(parents=True, exist_ok=True)
    stage_data_bp.parent.mkdir(parents=True, exist_ok=True)
    # Symlinks pointing to the per-window outputs (Windows-safe via copy)
    if hasattr(Path, "symlink_to"):
        try:
            stage_data_proc.symlink_to(proc_root.resolve(), target_is_directory=True)
            stage_data_bp.symlink_to(bp_dir.resolve(), target_is_directory=True)
        except (OSError, NotImplementedError):
            # Fallback to copy
            import shutil
            shutil.copytree(proc_root, stage_data_proc)
            shutil.copytree(bp_dir, stage_data_bp)
    else:
        import shutil
        shutil.copytree(proc_root, stage_data_proc)
        shutil.copytree(bp_dir, stage_data_bp)

    rob_dir.mkdir(parents=True, exist_ok=True)
    # v13: figures go to results/figures/sensitivity_savgol_window/<campaign>/window<N>/
    # while CSVs (clusters, bp_assignments, summaries) stay in rob_dir.
    from karst_analysis.io import resolve_figure_dir
    sensitivity_fig_root = resolve_figure_dir(
        "sensitivity_savgol_window", campaigns=[campaign],
    )
    fig_dir = sensitivity_fig_root / f"window{window:02d}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_clusters: list[pd.DataFrame] = []
    all_bp_assignments: list[pd.DataFrame] = []
    all_bic_summary: list[pd.DataFrame] = []
    all_sensitivity: list[pd.DataFrame] = []

    for well in WELLS.keys():
        try:
            res = compute_robustness(
                well, campaign=campaign,
                project_root=stage,
            )
        except ValueError:
            continue
        clusters = res.clusters.copy()
        clusters.insert(0, "well_id", well)
        clusters.insert(1, "delta_m", res.delta_m)
        clusters.insert(2, "savgol_window", window)
        all_clusters.append(clusters)
        all_bp_assignments.append(res.bp_records.assign(savgol_window=window))
        bic = res.bic_summary.copy()
        bic["savgol_window"] = window
        all_bic_summary.append(bic)

        sens_df = compute_robustness_sensitivity(
            well, campaign=campaign, project_root=stage,
        )
        if not sens_df.empty:
            sens_df["savgol_window"] = window
            all_sensitivity.append(sens_df)

        try:
            fig = plot_robustness_panel(
                res, well_id=well,
                output_path=fig_dir / f"robustness_{well}.png",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"  ⚠ figure error for {well}: {exc}")

    if all_clusters:
        pd.concat(all_clusters, ignore_index=True).to_csv(
            rob_dir / "robustness_clusters.csv", index=False,
        )
    if all_bp_assignments:
        pd.concat(all_bp_assignments, ignore_index=True).to_csv(
            rob_dir / "robustness_bp_assignments.csv", index=False,
        )
    if all_bic_summary:
        pd.concat(all_bic_summary, ignore_index=True).to_csv(
            rob_dir / "robustness_summary.csv", index=False,
        )
    if all_sensitivity:
        pd.concat(all_sensitivity, ignore_index=True).to_csv(
            rob_dir / "sensitivity_clusters.csv", index=False,
        )
    return rob_dir


def _make_comparison_figure(
    rob_dirs: dict[int, Path],
    *,
    output_path: Path,
    top_n: int = 6,
) -> None:
    """Render a comparison figure: top-N robust clusters per (well, window).

    One subplot per well (5 wells), x-axis = window, y-axis = depth.
    Each cluster appears as a horizontal segment showing depth_min..depth_max,
    with width proportional to its agreement score.
    """
    # Pool all clusters into one DataFrame with a window column
    parts = []
    for window, rob_dir in rob_dirs.items():
        f = rob_dir / "robustness_clusters.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        parts.append(df)
    if not parts:
        print("No robustness_clusters.csv found; nothing to plot.")
        return
    all_df = pd.concat(parts, ignore_index=True)

    wells = sorted(all_df["well_id"].unique())
    windows = sorted(rob_dirs.keys())

    n_wells = len(wells)
    fig, axes = plt.subplots(
        1, n_wells, figsize=(3.0 * n_wells, 7.0), sharey=True,
        gridspec_kw=dict(wspace=0.15),
    )
    if n_wells == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")
    n_w = len(windows)

    for ax, well in zip(axes, wells):
        sub = all_df[all_df["well_id"] == well]
        # Defensive: drop NaN/Inf in depth columns before computing axis limits.
        # (Can occur if a particular fit failed to converge for some N.)
        sub = sub[
            np.isfinite(sub["depth_min"]) & np.isfinite(sub["depth_max"])
        ].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {well}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(well, fontsize=10, fontweight="bold")
            ax.set_xticks(list(range(n_w)))
            ax.set_xticklabels([f"w={w}" for w in windows], fontsize=8)
            continue
        y_min = sub["depth_min"].min() - 1.0
        y_max = sub["depth_max"].max() + 1.0
        for i, window in enumerate(windows):
            sub_w = (sub[sub["savgol_window"] == window]
                       .head(top_n))
            color = cmap(i / max(n_w - 1, 1))
            for _, row in sub_w.iterrows():
                agreement = row["agreement"]
                # Bar width proportional to agreement (max 10)
                half_width = 0.35 * (agreement / 10.0)
                ax.fill_betweenx(
                    [row["depth_min"], row["depth_max"]],
                    i - half_width, i + half_width,
                    color=color, alpha=0.7, edgecolor=color,
                    linewidth=0.6,
                )
                # Median tick
                ax.plot([i - half_width, i + half_width],
                        [row["depth_median"], row["depth_median"]],
                        color="black", lw=0.8, alpha=0.85)
        ax.set_xticks(list(range(n_w)))
        ax.set_xticklabels([f"w={w}" for w in windows], fontsize=8)
        ax.set_xlabel("SavGol window", fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ax.set_title(well, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    axes[0].set_ylabel("Depth below ground level (m)", fontsize=10)

    fig.suptitle(
        f"SavGol window sensitivity — top-{top_n} robust clusters per well\n"
        f"(bar width ∝ agreement; median = thin black line)",
        fontsize=11, fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, left=0.07, right=0.98, bottom=0.10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Comparison figure: {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", required=True,
                   help="Folder with raw CSVs (e.g. data/raw/sec/2022_02).")
    p.add_argument("--campaign", required=True,
                   help="Campaign tag (e.g. 2022_02).")
    p.add_argument("--windows", nargs="+", type=int,
                   default=[7, 11, 15, 21, 31],
                   help="SavGol window values to test (must be odd).")
    p.add_argument("--output-base", default=None,
                   help="Output base dir "
                        "(default: results/sensitivity_savgol_window/<campaign>/).")
    args = p.parse_args()

    for w in args.windows:
        if w % 2 == 0:
            print(f"ERROR: window={w} is even; SavGol requires odd window.",
                  file=sys.stderr)
            return 2

    base_out = (Path(args.output_base)
                if args.output_base
                else Path("results") / SENSITIVITY_DIR_NAME / args.campaign)
    base_out.mkdir(parents=True, exist_ok=True)
    cfg_dir = base_out / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: raw-dir not found: {raw_dir}", file=sys.stderr)
        return 1

    print()
    print("=" * 72)
    print(" SAVGOL WINDOW SENSITIVITY SCAN")
    print("=" * 72)
    print(f"  raw-dir  : {raw_dir}")
    print(f"  campaign : {args.campaign}")
    print(f"  windows  : {args.windows}")
    print(f"  out-base : {base_out}")
    print("=" * 72)

    rob_dirs: dict[int, Path] = {}
    for w in args.windows:
        try:
            rob_dir = _run_pipeline_for_window(
                w, raw_dir=raw_dir, campaign=args.campaign,
                base_out=base_out, config_dir=cfg_dir,
            )
            rob_dirs[w] = rob_dir
            print(f"\n  ✓ window={w}: {rob_dir}")
        except Exception as exc:
            print(f"\n  ✗ window={w}: {exc}")

    if not rob_dirs:
        print("\nNo successful runs; nothing to compare.")
        return 1

    _make_comparison_figure(
        rob_dirs,
        output_path=base_out / "savgol_window_comparison.png",
    )

    print()
    print("=" * 72)
    print(" DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
