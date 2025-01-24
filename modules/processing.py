import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple

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


