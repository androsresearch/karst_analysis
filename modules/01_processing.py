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
        'Depth',
        'depth',
        'z',
        'Z'
    ],
    'conductivity': [
        'Corrected sp Cond [uS/cm]',
        'Corrected sp Cond [µS/cm]',
        'SpCond_muS/cm',
        'SEC',
        'Conductivity',
        'conductivity',
        'EC',
        'ec'
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


def apply_savgol_filter_to_df(df: pd.DataFrame,
                             value_col: Optional[str] = None,
                             window_length: int = 11,
                             poly_order: int = 3,
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
        smoothed_values = savgol_filter(df[value_col].values, 
                                       window_length=window_length, 
                                       polyorder=poly_order)
        result_df[value_col] = smoothed_values
        
        if logger:
            logger.info(f"Applied Savitzky-Golay filter (window={window_length}, order={poly_order})")
    else:
        if logger:
            logger.warning(f"Not enough data points ({len(df)}) for Savitzky-Golay filter (window={window_length})")
    
    return result_df


def process_borehole_data(df: pd.DataFrame,
                         apply_savgol: bool = False,
                         savgol_window: int = 11,
                         savgol_order: int = 3,
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
        'savgol_applied': False
    }
    
    # Step 1: Remove negative depths
    df_filtered = filter_non_negative_values(df, column_mappings=column_mappings, logger=logger)
    stats['negative_removed'] = stats['original_rows'] - len(df_filtered)
    
    # Step 2: Average duplicate depths
    df_averaged, duplicates = average_grouped_by_depth(df_filtered, column_mappings=column_mappings, logger=logger)
    stats['duplicates_found'] = len(duplicates)
    
    # Step 3: Resample to uniform spacing
    df_resampled = resample_profile_uniform(df_averaged, dz=dz, dz_method=dz_method, 
                                           column_mappings=column_mappings, logger=logger)
    
    # Step 4: Apply Savitzky-Golay filter (optional)
    if apply_savgol:
        df_final = apply_savgol_filter_to_df(df_resampled, window_length=savgol_window, 
                                            poly_order=savgol_order, 
                                            column_mappings=column_mappings, logger=logger)
        stats['savgol_applied'] = True
    else:
        df_final = df_resampled
    
    stats['final_rows'] = len(df_final)
    
    return df_final, stats