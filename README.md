# Mapping the Freshwater–Saltwater Interface in Karst  
**Linking Well Conductivity, ERT, and Satellite Observations in the Bahamas**


Copyright (c) 2025 Andros Research

All rights reserved.

This source code and associated documentation files are provided for viewing purposes only. No permission is granted to use, copy, reproduce, modify, merge, publish, distribute, sublicense, or sell any part of this code or its derivatives without explicit prior written permission from Andros Research.

Unauthorized use, reproduction, or distribution of this code is strictly prohibited and may result in legal action.


## Overview

This repository contains datasets, scripts, and supplementary materials associated with the research paper:  
**_Mapping the Freshwater–Saltwater Interface in Karst: Linking Well Conductivity, ERT, and Satellite Observations in the Bahamas_**

The study investigates how freshwater and saltwater interact in a coastal karst aquifer system. It focuses on understanding the shape and mixing behavior of the freshwater lens using a combination of field data and remote sensing.

## Study Focus

We analyze a well-defined transect from the center of a Bahamian island to the coast. The study aims to:

- Characterize the shape and thickness of the freshwater lens.
- Identify spatial variations in salinity.
- Understand the role of karst features (caves, fractures) in controlling mixing between fresh and salt water.
- Explore how satellite imagery and surface geomorphology relate to subsurface hydrology.

## Repository Structure
⚠️ Warning: This repository is currently under construction. File organization and content will change significantly as the project evolves.
```
📁 data/
├── wells/
│   ├── conductivity_profiles/
│   └── video_logs/
├── ert/
│   └── resistivity_profiles/
├── caves/
│   ├── maps/
│   └── water_profiles/
├── satellite/
│   └── surface_features/
└── historical/
    └── previous_profiles/
🔄 Note: ERT inversion data is currently hosted in a separate repository and will be integrated later: MGomezN/andros_resipy_inversions
```

```
📁 notebooks/
├── analysis_well_profiles.ipynb
├── compare_ert_wells.ipynb
├── satellite_overlay_analysis.ipynb
└── karst_mixing_zones.ipynb
```

```
📁 figures/
└── final_figures_for_paper/
```



## Data Sources

- **Wells**: AW5, AW6, BW3, LRS69, LRS70 (main transect), plus AW1–AW3, AW7, BW4 (adjacent control wells).
- **Caliper and Video Logs**: Identify cavities or fractures that could influence flow.
- **ERT (Electrical Resistivity Tomography)**: Provides lateral subsurface imaging to detect heterogeneities.
- **Cave Data**: Includes known inland cave systems and mixing profiles from sites like Uncle Charlie’s.
- **Satellite Data**: Includes surface texture, vegetation, dune ridges, and karst collapse polygons (shared by the Newfoundland team).

## Key Questions

- How does the freshwater lens shape and salinity profile change along the flow path?
- Is the interface between fresh and saltwater sharp or diffuse?
- Can observed salinity variations be explained by local karst connectivity?
- How well do satellite indicators predict subsurface karst features?

## Hypotheses

1. **Coastal Gradient Effects**  
   - 1A: Increased freshwater flow toward the coast.
   - 1B: Greater tidal mixing and saltwater intrusion near the coast.

2. **Karst Connectivity**  
   - Salinity variation depends on the borehole’s connection to cavity networks.
   - Differences in scale and geometry of cavities affect mixing behavior.

## Expected Outcomes

- High-resolution characterization of the freshwater–saltwater interface.
- Evaluation of how karst features influence subsurface flow and mixing.
- Integration of satellite-derived geomorphology with hydrogeological data.
- Insights applicable to other coastal karst aquifers.

# Installation

1. Clone the repository:
    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. Place all `.csv` files to be processed in the `data/row` folder.

6. Make the magic happen by running (from the project root):
    ```sh
    python main.py
    ```

> [!NOTE]  
> **Information:** It is possible to modify the Savitzky-Golay filter parameters as well as the segmented fitting and BIC parameters in `main.py`.

> [!WARNING]  
> **Attention:** Because of the level of calculations involved, processing all the data can take a considerable amount of time. Make sure everything is ready before running this command.

---

**Metrics and Charts**

To check the evaluation metrics (\(R^2\), adjusted \(R^2\), RMS, breakpoint locations) and visualize the results, run the `notebooks/evaluation.ipynb` notebook. The only required input is the file/profile name (without extension) that should be evaluated and displayed.  

An example profile (`BW5D_YSI_20230822`) is included in the data folder for testing the notebook.

---

**Boxplots**

1. Load all profiles in `data/rawdy` and the CSV file `fwl_2024_TW.csv` into `data/fwl_map`.
2. Run `notebooks/filter_statistics.ipynb`.
