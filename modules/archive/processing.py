import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from typing import Optional, List, Tuple
import os

def filter_non_negative_values(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the input arrays to include only non-negative values in `x`,
    removing corresponding elements in `y`.

    Parameters
    ----------
    x : np.ndarray
        Input x array to filter.
    y : np.ndarray
        Input y array to filter based on `x`.

    Returns
    -------
    x_filtered : np.ndarray
        Filtered x array containing only non-negative values.
    y_filtered : np.ndarray
        Filtered y array corresponding to the non-negative values in `x`.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    mask = x >= 0
    x_filtered = x[mask]
    y_filtered = y[mask]

    return x_filtered, y_filtered

def remove_outliers_iqr(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes outliers based on the IQR (Interquartile Range) of the values in `y`,
    applying the filter to both `x` and `y` arrays.

    Parameters
    ----------
    x : np.ndarray
        Array containing x-axis data (ordered or not).
    y : np.ndarray
        Array containing y-axis data.

    Returns
    -------
    x_filtered : np.ndarray
        Filtered x array without outliers.
    y_filtered : np.ndarray
        Filtered y array without outliers.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths or are empty.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Input arrays are empty.")

    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (y >= lower_bound) & (y <= upper_bound)
    x_filtered = x[mask]
    y_filtered = y[mask]

    return x_filtered, y_filtered

def average_grouped_by_x(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Groups `y` data by the unique values in `x`, calculating the average of `y`
    for each unique value in `x`.

    Parameters
    ----------
    x : np.ndarray
        Array containing x-axis data.
    y : np.ndarray
        Array containing y-axis data.

    Returns
    -------
    x_unique : np.ndarray
        Array with the unique values of `x`.
    y_mean : np.ndarray
        Array with the mean of `y` corresponding to each unique value of `x`.
    duplicated_frequencies : pd.DataFrame
        DataFrame containing duplicated values of `x` and their frequencies.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths or are empty.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Input arrays are empty.")

    # Sort by x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Identify indices where x changes value
    x_unique, group_indices, counts = np.unique(x_sorted, return_index=True, return_counts=True)

    # Find duplicated values and their frequencies
    duplicated_mask = counts > 1
    duplicated_values = x_unique[duplicated_mask]
    duplicated_frequencies = counts[duplicated_mask]

    # Create the DataFrame with duplicates and their frequencies
    duplicated_df = pd.DataFrame({
        'Duplicated Value': duplicated_values,
        'Frequency': duplicated_frequencies
    })

    # Calculate the mean of y for each unique group in x
    y_mean = np.add.reduceat(y_sorted, group_indices) / np.diff(np.append(group_indices, len(y_sorted)))

    return x_unique, y_mean, duplicated_df

def apply_savgol_filter(data: np.ndarray, window_length: int, poly_order: int) -> np.ndarray:
    """
    Applies the Savitzky-Golay filter to smooth a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data to be smoothed.
    window_length : int
        Length of the filter window. Must be a positive odd integer.
    poly_order : int
        Order of the polynomial. Must be less than `window_length`.

    Returns
    -------
    smoothed_data : np.ndarray
        Smoothed data.

    Raises
    ------
    ValueError
        If `window_length` is not a positive odd integer or if `poly_order` is greater than or equal to `window_length`.
    """
    if window_length % 2 == 0 or window_length <= 0:
        raise ValueError("The window length (window_length) must be a positive odd integer.")

    if poly_order >= window_length:
        raise ValueError("The polynomial order (poly_order) must be less than the window length (window_length).")

    # Apply the Savitzky-Golay filter
    return savgol_filter(data, window_length=window_length, polyorder=poly_order)

def filter_csv_by_vertical_position(
    df: pd.DataFrame,
    input_path: str,
    output_path: str,
    required_columns: Optional[List[str]] = None
) -> None:
    """
    Process multiple CSV files based on a reference vertical position and save the filtered data.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that must include at least the following columns:
            - 'name_file': str, the name of the CSV file to process.
            - 'chosen_vertical_position': numeric, the filter threshold for 'Vertical Position [m]'.
    input_path : str
        The directory where the input CSV files are located.
    output_path : str
        The directory where the filtered CSV files will be saved.
    required_columns : Optional[List[str]]
        A list of column names required in the CSV file. Defaults to 
        ['Vertical Position [m]', 'Corrected sp Cond [uS/cm]'] if not provided.
    """
    
    if required_columns is None:
        required_columns = ['Vertical Position [m]', 'Corrected sp Cond [uS/cm]']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for index, row in df.iterrows():
        file_name = row['name_file']
        chosen_vertical_position = row['chosen_vertical_position']
        
        input_file = os.path.join(input_path, file_name)
        
        if not os.path.isfile(input_file):
            print(f"[Warning] File '{file_name}' not found in '{input_path}'. Skipping this record.")
            continue
        
        try:
            data = pd.read_csv(input_file)
            
            if not set(required_columns).issubset(data.columns):
                print(f"[Warning] File '{file_name}' does not contain the required columns: {required_columns}.")
                continue
            
            data = data[required_columns]
            data_filtered = data[data[required_columns[0]] <= chosen_vertical_position]
            
            clean_name = file_name.replace("_rowdy.csv", "")
            clean_name = file_name.replace(".csv","")
            output_file = os.path.join(output_path, f'{clean_name}_filter.csv')
            
            data_filtered.to_csv(output_file, index=False)
            print(f"[Info] Processed and saved file: '{output_file}'")
            
        except Exception as e:
            print(f"[Error] An error occurred while processing file '{file_name}': {e}")


def analize_dz(depths, well_name,
               percentile=95):
    """
    Estimate the optimal sampling interval (dz) from a 1D depth array and produce a boxplot.

    Parameters
    ----------
    depths : array-like (1D)
        Depth measurements (in meters). Can be a pandas Series/DataFrame or numpy array.
    percentile : int, optional
        Percentile to use when estimating dz. Default is 95.
    well_name : str, optional
        Name of the well (for plot title). Default is "Well".

    Returns
    -------
    stats_df : pd.DataFrame
        Summary statistics for dz (percentile, median, mean, min, max, dz_max).
    fig : plotly.graph_objects.Figure
        Boxplot figure showing the distribution of Δz with metric lines.
    """
    # Coerce to 1D numpy array and drop NaNs
    if isinstance(depths, pd.DataFrame):
        depths = depths.iloc[:, 0]
    arr = np.asarray(depths).ravel()
    arr = arr[~np.isnan(arr)]

    sorted_depths = np.sort(arr)
    delta_z = np.diff(sorted_depths)

    # Compute summary statistics
    pval = np.percentile(delta_z, percentile)
    median = np.median(delta_z)
    mean = np.mean(delta_z)
    mn = np.min(delta_z)
    mx = np.max(delta_z)

    stats_df = pd.DataFrame({
        f'percentile{percentile}': [pval],
        'median': [median],
        'mean': [mean],
        'min': [mn],
        'max': [mx]
    })

    # Build boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=delta_z,
        name='Δz',
        boxpoints='outliers',
        marker_color='rgb(8,81,156)',
        line_color='rgb(8,81,156)'
    ))

    # Overlay horizontal lines for each metric
    metrics = {
        f'Percentile {percentile}': pval,
        'Median': median,
        'Mean': mean,
        'Min': mn,
        'Max': mx
    }
    colors = {
        f'Percentile {percentile}': 'red',
        'Median': 'green',
        'Mean': 'yellow',
        'Min': 'purple',
        'Max': 'orange'
    }
    for label, value in metrics.items():
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=colors[label],
            annotation_text=f"{label}: {value:.3f} m",
            annotation_position="right"
        )

    # Layout adjustments
    fig.update_layout(
        title=f'Distribution of sampling intervals (Δz): {well_name}',
        yaxis_title='Δz (meters)',
        showlegend=False,
        template='plotly_white',
        height=800,
        width=700,
        margin=dict(r=150)
    )

    return stats_df, fig

def resample_profile(
    df: pd.DataFrame,
    depth_col: str = 'Vertical Position m',
    ec_col: str = 'Corrected sp Cond [µS/cm]', 
    dz: float = None,
    dz_method: str = 'percentile95',
    adaptive_refinement: bool = False,
    sort_values: bool = False
) -> pd.DataFrame:
    """
    Resample a depth vs conductivity profile:
      1) Clean and optionally sort
      2) Create uniform depth grid
      3) Monotonic PCHIP interpolation
      4) Optional adaptive refinement in high-gradient zones

    Parameters
    ----------
    df : pandas.DataFrame
        Original data with depth and conductivity columns.
    depth_col : str
        Name of the depth column (meters).
    ec_col : str
        Name of the conductivity column (µS/cm).
    dz : float, optional
        Desired grid spacing. If None, calculated automatically.
    dz_method : str
        Method to estimate dz if not provided: 'percentile95', 'median', 'mean', 'min'.
    adaptive_refinement : bool
        If True, add midpoints in regions with steep gradients.
    sort_values : bool
        If True, sort data by depth before processing.

    Returns
    -------
    pandas.DataFrame or tuple
        If adaptive_refinement is False: DataFrame with columns:
          depth_col and ec_col sampled on uniform grid.
        If adaptive_refinement is True: tuple of:
          - full resampled DataFrame
          - DataFrame of added points (depth and conductivity)
    """
    # 1. Clean and sort
    data = df[[depth_col, ec_col]].dropna()
    if sort_values:
        data = data.sort_values(depth_col)

    depths = data[depth_col].values
    ecs = data[ec_col].values

    # 2. Determine dz_target
    if dz is None:
        dz_target = analize_dz(depths, percentile=95)[0]
        dz_target = float(dz_target[dz_method])
    else:
        dz_target = dz

    # 3. Build uniform depth grid including endpoints
    z_min, z_max = depths[0], depths[-1]
    uniform_grid = np.arange(z_min, z_max, dz_target)
    # Ensure z_max included
    if uniform_grid[-1] < z_max - 1e-10:
        uniform_grid = np.append(uniform_grid, z_max)

    # 4. PCHIP interpolation
    interpolator = PchipInterpolator(depths, ecs)
    ec_uniform = interpolator(uniform_grid)

    # 5. Adaptive refinement
    added_points = None
    if adaptive_refinement:
        # Compute absolute gradient
        grad = np.abs(np.gradient(ec_uniform, uniform_grid))
        threshold = 3 * np.median(grad)
        # Find indices where gradient exceeds threshold (excluding last index)
        high_grad_idx = np.where(grad > threshold)[0]
        # Only consider those with a next neighbor
        valid = high_grad_idx[high_grad_idx < len(uniform_grid) - 1]
        if valid.size > 0:
            # Compute midpoints
            mids = (uniform_grid[valid] + uniform_grid[valid + 1]) / 2.0
            # Unique and sorted
            mids = np.unique(mids)
            # Evaluate at mids
            ec_mids = interpolator(mids)
            # Merge grids
            final_grid = np.sort(np.concatenate([uniform_grid, mids]))
            ec_final = interpolator(final_grid)
            added_points = pd.DataFrame({depth_col: mids, ec_col: ec_mids})
        else:
            final_grid, ec_final = uniform_grid, ec_uniform
    else:
        final_grid, ec_final = uniform_grid, ec_uniform

    # 6. Package full result
    full_df = pd.DataFrame({depth_col: final_grid, ec_col: ec_final})

    if adaptive_refinement:
        return full_df, added_points
    
    return full_df
