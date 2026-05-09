"""Multi-technique convergence panels.

The convergence sub-package combines outputs from individual technique
sub-packages (sec, caliper, videolog, drilling, ert, satellite) into
single figures that highlight where multiple techniques agree on the
location of a feature (e.g. a karst cavity).

Available panels
----------------
caliper_video : caliper + per-sample severity bands + video-log notes +
                Ardaman lithology (when applicable). [v5.1]

sec_caliper_video : extends caliper_video with the SEC profile and its
                breakpoints overlaid on the caliper panel via a twin
                x-axis. Three columns: BP labels | caliper+SEC | video.
                [v6]

Available analyses
------------------
sec_caliper_match : quantitative SEC ↔ caliper convergence (Idea 3).
                Per-cluster matching of robust SEC clusters against
                anomalous caliper zones; binary and weighted scoring.
                Config-driven via ``convergence.sec_caliper`` block.

Future panels (planned):
    ert_convergence : add electrical resistivity tomography section.
"""

from karst_analysis.convergence.caliper_video import (
    WELLS, WellConfig, PanelConfig,
    plot_caliper_video_panel,
    build_all_caliper_video_panels,
)
from karst_analysis.convergence.sec_caliper_video import (
    SecCaliperVideoConfig,
    plot_sec_caliper_video_panel,
    build_all_sec_caliper_video_panels,
)
from karst_analysis.convergence.sec_caliper_match import (
    ConvergenceResult,
    compute_convergence,
    compute_overlap_m,
    compute_center_distance_m,
    find_matches,
    select_best_match,
    score_cluster,
    load_sec_clusters,
    load_caliper_zones,
)
from karst_analysis.convergence.sec_caliper_panel import (
    SecCaliperPanelConfig,
    plot_sec_caliper_panel,
    plot_master_panel,
    build_all_sec_caliper_panels,
)
from karst_analysis.convergence.site_panel import (
    SitePanelConfig,
    WELL_TYPE_LINESTYLE,
    DEFAULT_CAMPAIGN_PALETTE,
    plot_site_panel,
    plot_master_sites_panel,
    build_all_site_panels,
)
from karst_analysis.convergence.site_panel_interactive import (
    InteractiveSitePanelConfig,
    plot_site_panel_interactive,
    build_all_site_panels_interactive,
)

__all__ = [
    # Shared / caliper × video
    "WellConfig", "WELLS", "PanelConfig",
    "plot_caliper_video_panel",
    "build_all_caliper_video_panels",
    # SEC + caliper × video
    "SecCaliperVideoConfig",
    "plot_sec_caliper_video_panel",
    "build_all_sec_caliper_video_panels",
    # SEC ↔ caliper quantitative matching (Idea 3)
    "ConvergenceResult",
    "compute_convergence",
    "compute_overlap_m",
    "compute_center_distance_m",
    "find_matches",
    "select_best_match",
    "score_cluster",
    "load_sec_clusters",
    "load_caliper_zones",
    # SEC raw × caliper panel by WELL (v10/v11)
    "SecCaliperPanelConfig",
    "plot_sec_caliper_panel",
    "plot_master_panel",
    "build_all_sec_caliper_panels",
    # SEC raw × caliper panel by SITE (v12)
    "SitePanelConfig",
    "WELL_TYPE_LINESTYLE",
    "DEFAULT_CAMPAIGN_PALETTE",
    "plot_site_panel",
    "plot_master_sites_panel",
    "build_all_site_panels",
    # SEC raw × caliper panel by SITE — interactive HTML (v14)
    "InteractiveSitePanelConfig",
    "plot_site_panel_interactive",
    "build_all_site_panels_interactive",
]


