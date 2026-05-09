"""Depth/elevation reference transformations.

Different field instruments use different vertical references:
    - YSI conductivity probe : depth below water table (positive down)
    - Caliper                : depth below ground level (positive down)

To compare techniques in the same well, convert YSI depths to depth
below ground level by adding the vadose zone thickness.

Currently exposes:
    ysi_to_depth_below_ground
    load_well_metadata
    get_vadose_thickness
    extract_vadose_from_ysi_csv
"""

from karst_analysis.corrections.datum import (
    ysi_to_depth_below_ground,
    load_well_metadata,
    get_vadose_thickness,
    extract_vadose_from_ysi_csv,
)

__all__ = [
    "ysi_to_depth_below_ground",
    "load_well_metadata",
    "get_vadose_thickness",
    "extract_vadose_from_ysi_csv",
]
