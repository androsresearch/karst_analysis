# Usage of the Geophysical Survey Data Processing Script (`01_processing.py`)

This document describes how to use the `01_processing.py` script to process geophysical survey data from CSV files.

## General Description

The `01_processing.py` script is a command-line tool designed to clean, process, and standardize survey data. It reads CSV files from an input directory, performs various processing operations, and saves the results in an output directory.

The main features include:
- Reading multiple CSV files.
- Data cleaning (removal of negative values and duplicates).
- Optional application of a Savitzky-Golay smoothing filter.
- Generation of processed CSV files.
- Creation of a statistical summary of the processing.

## Requirements

To run this script, you need:
- **Python 3.x**
- All required dependencies can be installed from the requirements.txt file using:
  ```bash
  pip install -r requirements.txt
  ```
- The processing module `modules/01_processing.py`, which must be in the correct location relative to the main script.

## File Structure

The script expects a directory structure like the following:

```
.
├── data/
│   └── raw/
│       ├── sondeo_1.csv
│       └── sondeo_2.csv
├── docs/
├── modules/
│   └── 01_processing.py
└── scripts/
    └── 01_processing.py
```

- The raw CSV files should be in a directory (e.g., `data/raw`).
- The script is run from the project's root directory.

## How to Run the Script

1.  **Open a terminal** or command line.
2.  **Navigate to the root directory** of your project.
3.  **Run the script** with the following command:
    ```bash
    python scripts/01_processing.py
    ```

## Interactive Process

When you run the script, an interactive process will start to guide you through the processing setup:

1.  **Input path**: You will need to provide the relative path to the directory containing the raw CSV files (e.g., `data/raw`). The script will check if the folder exists and contains CSV files.

2.  **Output path**: You will need to provide the relative path to the folder where the processed files will be saved (e.g., `data/processed`). If the folder does not exist, the script will create it.

3.  **Savitzky-Golay filter**: You will be asked if you want to apply a smoothing filter.
    - Enter `y` (yes) or `n` (no). The default value is `n`.
    - If you choose `y`, you will be asked for two additional parameters:
        - **Window length**: An odd number for the filter window (default: 11).
        - **Polynomial order**: The order of the polynomial to fit (default: 3).

## Script Output

Once the process is finished, you will find the following files in the output folder you specified:

- **Processed CSV files**: For each input file `[name].csv`, a `[name]_processed.csv` will be generated. These files contain the cleaned and, if applied, smoothed data.
- **Console log**: During execution, you will see progress and results in real time.

## Processing Summary

At the end, the script will print a detailed summary in the console, which includes:

- **General statistics**:
    - Total number of files processed.
    - Number of files successfully processed and with errors.
    - Total processing time.
- **Statistics summary (for successful files)**:
    - Total original and final rows.
    - Number of negative values removed and duplicates found.
    - Percentage of data reduction.
- **Failed files**: If any file could not be processed, it will be listed along with the corresponding error. 