"""karst_analysis — multi-method analysis of a coastal karst aquifer.

Sub-packages (current):
    sec : Specific Electrical Conductivity profiles (active, ★ priority).

Sub-packages (stubs, future work):
    caliper, videolog, ert, satellite : technique-specific modules.
    convergence : cross-technique integration (the central scientific
        contribution; uses outputs from the technique sub-packages).

Shared utilities:
    io          : generic loaders and filename parsing.
    corrections : depth/elevation reference transformations.
    metadata    : well metadata access.
    runs        : run tracking and reproducibility helpers.
"""

__version__ = "0.1.0"
