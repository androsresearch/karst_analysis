import os
import random
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# Utility functions for file name handling and data simulation
# =============================================================================
def get_file_suffix(subfolder: str) -> str:
    """
    Determine the file suffix (or extension) based on the subfolder.
    
    Args:
        subfolder (str): Name of the subfolder ('rawdy/', 'processed/', or 'raw/').
        
    Returns:
        str: Suffix to be appended to the base file name.
    """
    mapping = {
        'rawdy': '_rowdy.csv',
        'processed': '_processed.csv',
        'raw': '.csv'
    }
    key = subfolder.strip('/').lower()
    if key not in mapping:
        raise ValueError(f"Invalid subfolder name: {subfolder}")
    return mapping[key]


# =============================================================================
# Data Filtering Module
# =============================================================================
def filter_well_data(well_data: pd.DataFrame, filter_value: float) -> pd.DataFrame:
    """
    Filter the well data based on the vertical position criterion.
    Keeps only rows where 'Vertical Position [m]' is less than or equal to filter_value.
    
    Args:
        well_data (pd.DataFrame): DataFrame with well data.
        filter_value (float): The threshold value for filtering.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if "Vertical Position [m]" not in well_data.columns:
        raise KeyError("Column 'Vertical Position [m]' not found in the data.")
    filtered = well_data[well_data["Vertical Position [m]"] <= filter_value].copy()
    return filtered


def load_and_filter_data(
    file_info_df: pd.DataFrame, 
    subfolder: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load and filter data for each well based on the three filtering methods.
    
    The file_info_df must contain the columns:
        - 'ID'
        - 'vp_dgh': Filter value for method DGH.
        - 'vp_bic': Filter value for method BIC.
        - 'vp_ic': Filter value for method IC.
        
    Args:
        file_info_df (pd.DataFrame): DataFrame with filtering points for each well.
        subfolder (str): The subfolder name (e.g. 'rawdy', 'processed', or 'raw').
        
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary where the first key is the well ID 
        and the second key is the filtering method (e.g. 'IC', 'BIC', 'DGH'), and the value is 
        the filtered DataFrame.
    """
    filtered_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    # Map filtering method keys to the corresponding column in file_info_df
    method_mapping = {
        'IC': 'vp_ic',
        'BIC': 'vp_bic',
        'DGH': 'vp_dgh'
    }

    root = os.path.abspath('..')  
    
    for idx, row in file_info_df.iterrows():
        well_id = row['ID']
        # Read data (simulate reading CSV based on the subfolder)
        try:
            df = pd.read_csv(f"{root}/data/{subfolder}/{well_id}{get_file_suffix(subfolder)}")
        except Exception as e:
            print(f"Error reading data for {well_id}: {e}")
            continue
        
        filtered_data[well_id] = {}
        for method, col_name in method_mapping.items():
            if col_name not in row:
                raise KeyError(f"Filtering column '{col_name}' not found in file_info_df.")
            filter_value = row[col_name]
            filtered_df = filter_well_data(df, filter_value)
            filtered_data[well_id][method] = filtered_df
    
    return filtered_data


# =============================================================================
# Boxplot Generation Module
# =============================================================================
def calculate_outliers(data: np.ndarray) -> int:
    """
    Calculate the number of outliers in the data using the IQR method.
    
    Args:
        data (np.ndarray): 1D array of numerical values.
    
    Returns:
        int: Number of outliers.
    """
    if data.size == 0:
        return 0
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return int(outliers)

def generate_boxplots(
    filtered_data: Dict[str, Dict[str, pd.DataFrame]],
    variable: str,
    show_outliers: bool = True,
    order: Optional[List[str]] = None,
    legend_x: float = 1.02,
    legend_y: float = 1.0,
    legend_orientation: str = 'v'
) -> go.Figure:
    """
    Generate boxplots (horizontal) with Plotly for each well and each filtering method.
    The legend groups and labels are set by method (e.g., 'IC', 'BIC', 'DGH'), 
    and each method is assigned a unique color.

    Args:
        filtered_data (Dict[str, Dict[str, pd.DataFrame]]): 
            Nested dictionary with filtered data. 
            The first key is the well ID, 
            and the second key is the filtering method ('IC', 'BIC', 'DGH', etc.).
        variable (str): 
            Name of the column to plot in the boxplots (e.g., "Corrected sp Cond [uS/cm]" or 
            "Vertical Position [m]").
        show_outliers (bool): 
            Whether to display outliers in the boxplots.
        order (Optional[List[str]]): 
            Optional list that defines the order of the boxplots on the y-axis. 
            Each entry should have the format "WellID - Method".
        legend_x (float): 
            X position of the legend.
        legend_y (float): 
            Y position of the legend.
        legend_orientation (str):
            Orientation of the legend: 'v' for vertical, 'h' for horizontal.

    Returns:
        go.Figure: 
            A Plotly Figure object containing the boxplots and annotations.
    """
    
    # Dictionary to assign a specific color to each method
    method_colors = {
        'BIC': 'blue',
        'IC': 'red',
        'DGH': 'green'
    }
    
    # Lists for labeling and data
    group_labels = []
    group_data = []
    method_list = []
    
    # Extract data from the nested dictionary
    for well_id, methods_dict in filtered_data.items():
        for method, df in methods_dict.items():
            # "label" is used for the y-axis category
            label = f"{well_id} - {method}"
            group_labels.append(label)
            group_data.append(df[variable].values)
            method_list.append(method)
    
    # Optional ordering of the categories on the y-axis
    if order:
        # Create a lookup for the desired index of each label
        order_dict = {val: idx for idx, val in enumerate(order)}
        combined = list(zip(group_labels, group_data, method_list))
        
        # Sort using the 'order' list
        combined_sorted = sorted(
            combined,
            key=lambda x: order_dict.get(x[0], float('inf'))
        )
        
        group_labels, group_data, method_list = zip(*combined_sorted)
        group_labels, group_data, method_list = (
            list(group_labels), list(group_data), list(method_list)
        )
    
    # Decide how to display outliers
    boxpoints_setting: Any = "outliers" if show_outliers else False
    
    # We keep track of which methods we have already used, so we don't repeat them in the legend
    used_methods = set()
    box_traces = []
    
    # Create boxplot traces
    for label, data_array, method in zip(group_labels, group_data, method_list):
        # Use the method name to look up a color; default to 'gray' if not found
        color = method_colors.get(method, 'gray')
        
        # name: what appears in the legend
        # y: the category for the horizontal boxes
        trace = go.Box(
            x=data_array,
            y=[label] * len(data_array),  # This defines the category on the y-axis
            name=method,                  # This defines the legend label
            boxpoints=boxpoints_setting,
            orientation='h',
            marker_color=color,
            legendgroup=method,           # Group by method
            showlegend=(method not in used_methods)
        )
        
        box_traces.append(trace)
        used_methods.add(method)
    
    # Helper function to count outliers using the IQR method
    def calculate_outliers(data: np.ndarray) -> int:
        if data.size == 0:
            return 0
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((data < lower_bound) | (data > upper_bound))
        return int(outliers)
    
    # Create annotations for number of points and outliers
    annotations = []
    for label, data_array in zip(group_labels, group_data):
        n_points = len(data_array)
        n_outliers = calculate_outliers(data_array)
        annotation_text = f"n={n_points}, out={n_outliers}"
        
        annotation = dict(
            x=1.15,    # position outside main plot area
            y=label,
            xref="paper",
            yref="y",
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
        annotations.append(annotation)
    
    # Build the figure
    fig = go.Figure(data=box_traces)
    
    fig.update_layout(
        yaxis=dict(
            title="Well - Filtering Method",
            # Ensure categories are shown in the correct order
            categoryorder="array",
            categoryarray=group_labels
        ),
        xaxis=dict(
            title=variable
        ),
        margin=dict(r=150),
        annotations=annotations,
        template="plotly_white",
        title=f"Boxplots of {variable} by Well and Filtering Method",
        legend=dict(
            x=legend_x,
            y=legend_y,
            orientation=legend_orientation
        )
    )
    
    return fig

