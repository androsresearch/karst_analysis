import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from typing import Optional, List, Tuple, Dict, Union
import logging

# Default column name mappings
DEFAULT_COLUMN_MAPPINGS = {
    'depth': [
        'Vertical Position [m]',
        'Vertical Position m',
        'VP',
        'z',
        'Z'
    ],
    'conductivity': [
        'Corrected sp Cond [uS/cm]',
        'Corrected sp Cond [µS/cm]',
        'SpCond_muS/cm',
        'SpCond µS/cm',
        'SEC',
        'Conductivity',
        'conductivity',
        'EC',
        'ec'
    ],
    'time': [
        'Time (HH:mm:ss)',
    ],
    'time_frac': [
        'Time (Fract. Sec)',
    ]
}


def find_column_name(df: pd.DataFrame, column_type: str, 
                     column_mappings: Dict[str, List[str]] = None) -> Optional[str]:
    """
    Find the actual column name in a DataFrame based on possible variations.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search in.
    column_type : str
        The type of column to find ('depth' or 'conductivity').
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings. If None, uses DEFAULT_COLUMN_MAPPINGS.
        
    Returns
    -------
    Optional[str]
        The found column name, or None if not found.
    """
    if column_mappings is None:
        column_mappings = DEFAULT_COLUMN_MAPPINGS
        
    possible_names = column_mappings.get(column_type, [])
    
    for name in possible_names:
        if name in df.columns:
            return name
            
    # Case-insensitive search as fallback
    df_columns_lower = [col.lower() for col in df.columns]
    for name in possible_names:
        if name.lower() in df_columns_lower:
            idx = df_columns_lower.index(name.lower())
            return df.columns[idx]
    
    return None


def ensure_chronological_order(df: pd.DataFrame,
                               column_mappings: Optional[Dict[str, List[str]]] = None,
                               logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Ensure the DataFrame is ordered chronologically by time columns if they exist.
    """
    time_col = find_column_name(df, 'time', column_mappings)
    time_frac_col = find_column_name(df, 'time_frac', column_mappings)
    
    sort_cols = []
    if time_col:
        sort_cols.append(time_col)
    if time_frac_col:
        sort_cols.append(time_frac_col)
        
    if sort_cols:
        return df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        if logger:
            logger.warning("Could not find time columns. Returning unsorted DataFrame copy.")
        return df.copy()


def filter_monotonic_descent(df: pd.DataFrame,
                             depth_col: Optional[str] = None,
                             tolerance: float = 0.002,
                             column_mappings: Optional[Dict[str, List[str]]] = None,
                             logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter out depth readings that violate monotonic descent (e.g. probe bouncing).
    """
    if depth_col is None:
        depth_col = find_column_name(df, 'depth', column_mappings)
        if depth_col is None:
            raise ValueError("Could not find depth column in DataFrame")
            
    depths = df[depth_col].values
    keep_mask = np.ones(len(df), dtype=bool)
    
    max_depth = -np.inf
    max_reversal = 0.0
    
    for i, d in enumerate(depths):
        if d >= max_depth - tolerance:
            max_depth = max(max_depth, d)
        else:
            keep_mask[i] = False
            max_reversal = max(max_reversal, max_depth - d)
            
    df_filtered = df[keep_mask].copy()
    total = len(df)
    kept = keep_mask.sum()
    removed = total - kept
    
    stats = {
        'total_readings': total,
        'kept_readings': kept,
        'removed_readings': removed,
        'removal_pct': (removed / total * 100) if total > 0 else 0,
        'max_reversal_m': max_reversal
    }
    
    if logger and removed > 0:
        logger.info(f"Monotonic filter removed {removed} readings ({stats['removal_pct']:.1f}%). Max reversal: {max_reversal:.3f}m")
        
    return df_filtered, stats


def filter_non_negative_values(df: pd.DataFrame, 
                               depth_col: Optional[str] = None,
                               value_col: Optional[str] = None,
                               column_mappings: Optional[Dict[str, List[str]]] = None,
                               logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Filter DataFrame to include only non-negative depth values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with depth and value columns.
    depth_col : str, optional
        Name of the depth column. If None, will be auto-detected.
    value_col : str, optional
        Name of the value column. If None, will be auto-detected.
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only non-negative depth values.
    """
    # Auto-detect columns if not specified
    if depth_col is None:
        depth_col = find_column_name(df, 'depth', column_mappings)
        if depth_col is None:
            raise ValueError("Could not find depth column in DataFrame")
            
    if value_col is None:
        value_col = find_column_name(df, 'conductivity', column_mappings)
        if value_col is None:
            raise ValueError("Could not find conductivity column in DataFrame")
    
    original_len = len(df)
    filtered_df = df[df[depth_col] >= 0].copy()
    removed_count = original_len - len(filtered_df)
    
    if logger:
        logger.info(f"Removed {removed_count} rows with negative depth values")
    
    return filtered_df


def average_grouped_by_depth(df: pd.DataFrame,
                            depth_col: Optional[str] = None,
                            value_col: Optional[str] = None,
                            column_mappings: Optional[Dict[str, List[str]]] = None,
                            logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group data by unique depth values and average the corresponding values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with depth and value columns.
    depth_col : str, optional
        Name of the depth column. If None, will be auto-detected.
    value_col : str, optional
        Name of the value column. If None, will be auto-detected.
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with unique depths and averaged values
        - DataFrame with duplicate information (depths and frequencies)
    """
    # Auto-detect columns if not specified
    if depth_col is None:
        depth_col = find_column_name(df, 'depth', column_mappings)
        if depth_col is None:
            raise ValueError("Could not find depth column in DataFrame")
            
    if value_col is None:
        value_col = find_column_name(df, 'conductivity', column_mappings)
        if value_col is None:
            raise ValueError("Could not find conductivity column in DataFrame")
    
    # Group by depth and calculate mean
    grouped = df.groupby(depth_col)[value_col].agg(['mean', 'count']).reset_index()
    
    # Find duplicates
    duplicates = grouped[grouped['count'] > 1][[depth_col, 'count']].copy()
    duplicates.columns = ['Duplicated Depth', 'Frequency']
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        depth_col: grouped[depth_col],
        value_col: grouped['mean']
    })
    
    if logger and len(duplicates) > 0:
        logger.info(f"Found {len(duplicates)} duplicate depth values, averaged their conductivity")
    
    return result_df, duplicates


def resample_profile_uniform(df: pd.DataFrame,
                           depth_col: Optional[str] = None,
                           value_col: Optional[str] = None,
                           dz: Optional[float] = None,
                           dz_method: str = 'percentile95',
                           column_mappings: Optional[Dict[str, List[str]]] = None,
                           logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Resample depth profile to uniform spacing using PCHIP interpolation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with depth and value columns.
    depth_col : str, optional
        Name of the depth column. If None, will be auto-detected.
    value_col : str, optional
        Name of the value column. If None, will be auto-detected.
    dz : float, optional
        Target spacing. If None, calculated automatically.
    dz_method : str
        Method to calculate dz: 'percentile95', 'median', 'mean', 'min'.
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with uniform depth spacing.
    """
    # Auto-detect columns if not specified
    if depth_col is None:
        depth_col = find_column_name(df, 'depth', column_mappings)
        if depth_col is None:
            raise ValueError("Could not find depth column in DataFrame")
            
    if value_col is None:
        value_col = find_column_name(df, 'conductivity', column_mappings)
        if value_col is None:
            raise ValueError("Could not find conductivity column in DataFrame")
    
    # Sort by depth
    df_sorted = df.sort_values(depth_col).copy()
    depths = df_sorted[depth_col].values
    values = df_sorted[value_col].values
    
    # Calculate dz if not provided
    if dz is None:
        delta_z = np.diff(depths)
        if dz_method == 'percentile95':
            dz = np.percentile(delta_z, 95)
        elif dz_method == 'median':
            dz = np.median(delta_z)
        elif dz_method == 'mean':
            dz = np.mean(delta_z)
        elif dz_method == 'min':
            dz = np.min(delta_z)
        else:
            raise ValueError(f"Unknown dz_method: {dz_method}")
    
    # Create uniform grid
    z_min, z_max = depths.min(), depths.max()
    uniform_grid = np.arange(z_min, z_max + dz, dz)
    
    # Ensure we don't exceed original bounds
    uniform_grid = uniform_grid[uniform_grid <= z_max]
    
    # PCHIP interpolation
    interpolator = PchipInterpolator(depths, values)
    values_uniform = interpolator(uniform_grid)
    
    if logger:
        logger.info(f"Resampled from {len(df)} to {len(uniform_grid)} points with dz={dz:.4f}")
    
    return pd.DataFrame({
        depth_col: uniform_grid,
        value_col: values_uniform
    })


def _savgol_segmented(values: np.ndarray,
                      window_length: int = 11,
                      poly_order: int = 3,
                      gradient_factor: float = 20.0,
                      min_gradient_threshold: float = 1000.0,
                      logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Apply Savitzky-Golay filter segment by segment to preserve transition points.
    """
    gradients = np.abs(np.diff(values))
    threshold = max(np.median(gradients) * gradient_factor, min_gradient_threshold)
    
    discontinuities = np.where(gradients > threshold)[0]
    
    if len(discontinuities) == 0:
        return savgol_filter(values, window_length=window_length, polyorder=poly_order)
        
    if logger:
        logger.info(f"Detected {len(discontinuities)} gradient discontinuities")
        
    half_window = window_length // 2
    is_transition = np.zeros(len(values), dtype=bool)
    
    for idx in discontinuities:
        start_idx = max(0, idx - half_window)
        end_idx = min(len(values), idx + half_window + 2)
        is_transition[start_idx:end_idx] = True
        
    result = values.copy()
    
    # Identify segments
    padded = np.concatenate(([True], is_transition, [True]))
    transitions = np.where(padded[:-1] != padded[1:])[0]
    
    segments = []
    for i in range(0, len(transitions), 2):
        start = transitions[i]
        end = transitions[i+1]
        segments.append((start, end))
        
    smoothed_count = 0
    for start, end in segments:
        length = end - start
        if length >= window_length:
            result[start:end] = savgol_filter(values[start:end], window_length=window_length, polyorder=poly_order)
            smoothed_count += 1
            
    preserved_count = is_transition.sum()
    if logger:
        logger.info(f"Smoothed {smoothed_count} segments; {preserved_count} transition points preserved")
        
    return result


def apply_savgol_filter_to_df(df: pd.DataFrame,
                             value_col: Optional[str] = None,
                             window_length: int = 11,
                             poly_order: int = 3,
                             segmented: bool = True,
                             gradient_factor: float = 20.0,
                             min_gradient_threshold: float = 1000.0,
                             column_mappings: Optional[Dict[str, List[str]]] = None,
                             logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to smooth the conductivity values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with value column to smooth.
    value_col : str, optional
        Name of the value column. If None, will be auto-detected.
    window_length : int
        Length of the filter window (must be odd).
    poly_order : int
        Order of the polynomial (must be less than window_length).
    segmented : bool
        Whether to use segmented Savitzky-Golay filtering to preserve step changes.
    gradient_factor : float
        Factor to multiply median gradient to determine discontinuity threshold.
    min_gradient_threshold : float
        Minimum absolute change to be considered a discontinuity.
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed values.
    """
    # Auto-detect column if not specified
    if value_col is None:
        value_col = find_column_name(df, 'conductivity', column_mappings)
        if value_col is None:
            raise ValueError("Could not find conductivity column in DataFrame")
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
        if logger:
            logger.warning(f"window_length must be odd, adjusted to {window_length}")
    
    # Apply filter if we have enough points
    if len(df) >= window_length:
        if segmented:
            smoothed_values = _savgol_segmented(
                result_df[value_col].values,
                window_length=window_length,
                poly_order=poly_order,
                gradient_factor=gradient_factor,
                min_gradient_threshold=min_gradient_threshold,
                logger=logger
            )
        else:
            smoothed_values = savgol_filter(
                result_df[value_col].values, 
                window_length=window_length, 
                polyorder=poly_order
            )
            if logger:
                logger.info(f"Applied Savitzky-Golay filter (window={window_length}, order={poly_order})")
                
        result_df[value_col] = smoothed_values
    else:
        if logger:
            logger.warning(f"Not enough data points ({len(df)}) for Savitzky-Golay filter (window={window_length})")
    
    return result_df


def adjust_vertical_position(df: pd.DataFrame,
                           depth_col: Optional[str] = None,
                           adjustment: float = 0.272,
                           method: str = 'TOM',
                           column_mappings: Optional[Dict[str, List[str]]] = None,
                           logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Adjust vertical position values by adding an offset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with depth column.
    depth_col : str, optional
        Name of the depth column. If None, will be auto-detected.
    adjustment : float, default 0.272
        Value to add to vertical positions.
    method : str, default 'TOM'
        Adjustment method:
        - 'TOM': Add adjustment only to values >= 0.001 m, leave values <= 0.00 m unchanged
        - 'YSI': Add adjustment to all values
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted vertical positions.
        
    Raises
    ------
    ValueError
        If depth column cannot be found or method is invalid.
    """
    # Auto-detect column if not specified
    if depth_col is None:
        depth_col = find_column_name(df, 'depth', column_mappings)
        if depth_col is None:
            raise ValueError("Could not find depth column in DataFrame")
    
    # Validate method
    valid_methods = ['TOM', 'YSI']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    if method == 'TOM':
        # TOM method: only adjust values >= 0.001 m
        mask = result_df[depth_col] >= 0.001
        adjusted_count = mask.sum()
        unchanged_count = (~mask).sum()
        
        result_df.loc[mask, depth_col] = result_df.loc[mask, depth_col] + adjustment
        
        if logger:
            logger.info(f"Applied TOM adjustment (+{adjustment} m): "
                       f"{adjusted_count} values adjusted, {unchanged_count} values unchanged")
    
    elif method == 'YSI':
        # YSI method: adjust all values
        result_df[depth_col] = result_df[depth_col] + adjustment
        
        if logger:
            logger.info(f"Applied YSI adjustment (+{adjustment} m) to all {len(result_df)} values")
    
    return result_df


def process_borehole_data(df: pd.DataFrame,
                         apply_savgol: bool = False,
                         savgol_window: int = 11,
                         savgol_order: int = 3,
                         savgol_segmented: bool = True,
                         apply_depth_adjustment: bool = False,
                         depth_adjustment: float = 0.272,
                         depth_adjustment_method: str = 'TOM',
                         apply_monotonic_filter: bool = True,
                         monotonic_tolerance: float = 0.002,
                         dz: Optional[float] = None,
                         dz_method: str = 'percentile95',
                         column_mappings: Optional[Dict[str, List[str]]] = None,
                         logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply complete processing pipeline to borehole data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with depth and conductivity data.
    apply_savgol : bool
        Whether to apply Savitzky-Golay filter.
    savgol_window : int
        Window length for Savitzky-Golay filter.
    savgol_order : int
        Polynomial order for Savitzky-Golay filter.
    savgol_segmented : bool
        Whether to use segmented Savitzky-Golay filtering.
    apply_depth_adjustment : bool
        Whether to apply depth adjustment.
    depth_adjustment : float
        Value to add to depth positions.
    depth_adjustment_method : str
        Method for depth adjustment ('TOM' or 'YSI').
    apply_monotonic_filter : bool
        Whether to filter out reversed depth readings (probe bounce).
    monotonic_tolerance : float
        Tolerance for monotonic filter.
    dz : float, optional
        Target spacing for resampling.
    dz_method : str
        Method to calculate dz if not provided.
    column_mappings : Dict[str, List[str]], optional
        Custom column name mappings.
    logger : logging.Logger, optional
        Logger for output messages.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - Processed DataFrame
        - Dictionary with processing statistics
    """
    stats = {
        'original_rows': len(df),
        'negative_removed': 0,
        'duplicates_found': 0,
        'final_rows': 0,
        'savgol_applied': False,
        'savgol_segmented': savgol_segmented,
        'depth_adjustment_applied': False,
        'depth_adjustment_method': None,
        'monotonic_filter_applied': False,
        'monotonic_removed': 0,
        'monotonic_removal_pct': 0.0,
    }
    
    # Step 1: Ensure chronological order
    df = ensure_chronological_order(df, column_mappings=column_mappings, logger=logger)
    
    # Step 2: Apply depth adjustment (if requested)
    if apply_depth_adjustment:
        df = adjust_vertical_position(df, 
                                    adjustment=depth_adjustment,
                                    method=depth_adjustment_method,
                                    column_mappings=column_mappings, 
                                    logger=logger)
        stats['depth_adjustment_applied'] = True
        stats['depth_adjustment_method'] = depth_adjustment_method
    
    # Step 3: Remove negative depths
    df_filtered = filter_non_negative_values(df, column_mappings=column_mappings, logger=logger)
    stats['negative_removed'] = stats['original_rows'] - len(df_filtered)
    
    # Step 4: monotonic filter
    if apply_monotonic_filter:
        df_filtered, mono_stats = filter_monotonic_descent(
            df_filtered,
            tolerance=monotonic_tolerance,
            column_mappings=column_mappings,
            logger=logger
        )
        stats['monotonic_filter_applied'] = True
        stats['monotonic_removed'] = mono_stats['removed_readings']
        stats['monotonic_removal_pct'] = mono_stats['removal_pct']
    
    # Step 5: Average duplicate depths
    df_averaged, duplicates = average_grouped_by_depth(df_filtered, column_mappings=column_mappings, logger=logger)
    stats['duplicates_found'] = len(duplicates)
    
    # Step 6: Resample to uniform spacing
    df_resampled = resample_profile_uniform(df_averaged, dz=dz, dz_method=dz_method, 
                                           column_mappings=column_mappings, logger=logger)
    
    # Step 7: Apply Savitzky-Golay filter (optional)
    if apply_savgol:
        df_final = apply_savgol_filter_to_df(df_resampled, window_length=savgol_window, 
                                            poly_order=savgol_order, 
                                            segmented=savgol_segmented,
                                            column_mappings=column_mappings, logger=logger)
        stats['savgol_applied'] = True
    else:
        df_final = df_resampled
    
    stats['final_rows'] = len(df_final)
    
    return df_final, stats
