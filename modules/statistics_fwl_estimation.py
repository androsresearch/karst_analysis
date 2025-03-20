import os
import random
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
# Statistical Analysis Module
# =============================================================================

def compute_statistics(filtered_data: dict, file_info_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Compute descriptive statistics for each well and method based on the specified column,
    and merge the resulting DataFrame with vertical position values from `file_info_df`.

    Parameters:
    -----------
    filtered_data : dict
        A nested dictionary where keys are well IDs (base name of the well) and values are
        dictionaries with method names ('IC', 'BIC', 'DGH') as keys and DataFrames as values.
    file_info_df : pd.DataFrame
        A DataFrame containing vertical position values for each well and method.
    column : str
        The column for which to compute statistics. Must be one of:
            - "Vertical Position [m]"
            - "Corrected sp Cond [uS/cm]"

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the following columns:
         - ID: Base name of the well
         - Method: Filtering method used (e.g. IC, BIC, DGH)
         - cv: Coefficient of Variation (std / mean)
         - iqr: Interquartile Range (75th percentile - 25th percentile)
         - vp_selected: The vertical position selected for the given method.
    """

    stats_list = []

    # Iterate over each well in the filtered_data dictionary
    for well_id, methods in filtered_data.items():
        # Get corresponding row in file_info_df based on well_id
        vp_row = file_info_df[file_info_df["ID"] == well_id]

        # Ensure the well exists in the file_info_df
        if vp_row.empty:
            continue

        for method, df in methods.items():
            if column not in df.columns:
                continue  # Skip if the column is not available
            
            # Extract the column and drop missing values
            data_series = df[column].dropna()
            if data_series.empty:
                continue  # Skip if no data is present after filtering

            # Compute statistics
            mean_val = data_series.mean()
            std_val = data_series.std()
            cv_val = std_val / mean_val if mean_val != 0 else float('nan')
            min_val = data_series.min()
            max_val = data_series.max()
            median_val = data_series.median()
            percentile_25 = data_series.quantile(0.25)
            percentile_50 = data_series.quantile(0.50)  # equivalent to median
            percentile_75 = data_series.quantile(0.75)
            iqr_val = percentile_75 - percentile_25

            # Assign vertical position based on method
            if method == "DGH":
                vp_selected = vp_row["vp_dgh"].values[0]
            elif method == "BIC":
                vp_selected = vp_row["vp_bic"].values[0]
            elif method == "IC":
                vp_selected = vp_row["vp_ic"].values[0]
            else:
                vp_selected = None  # Just in case an unknown method appears

            # Append results
            stats_list.append({
                "id": well_id,
                "method": method,
                "fwl_thickness": vp_selected,
                "mean": mean_val,
                "std": std_val,
                "cv": cv_val,
                "min": min_val,
                "max": max_val,
                "median": median_val,
                "25%": percentile_25,
                "50%": percentile_50,
                "75%": percentile_75,
                "iqr": iqr_val
            })

    # Convert the list to a DataFrame
    stats_df = pd.DataFrame(stats_list)

    return stats_df


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

def generate_boxplots_matplotlib(
    filtered_data: Dict[str, Dict[str, pd.DataFrame]],
    variable: str,
    show_outliers: bool = True,
    order: Optional[List[str]] = None,
    legend_x: float = 1.02,
    legend_y: float = 1.0,
    legend_orientation: str = 'vertical',
    mirror_top_axis: bool = False,
    enable_minor_ticks: bool = False,
    draw_vertical_line: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates horizontal boxplots with Matplotlib for each well and each filtering method.
    Optionally:
        - Adds a mirrored X-axis on the top (mirror_top_axis).
        - Displays minor ticks every 1000 units (enable_minor_ticks).
        - Draws a dotted vertical line at x=1452.9 (draw_vertical_line).

    Args:
        filtered_data (Dict[str, Dict[str, pd.DataFrame]]):
            Nested dictionary containing the filtered data.
            - First key: Well ID
            - Second key: Filtering method (e.g., 'IC', 'BIC', 'DGH').
        variable (str):
            Name of the column to be plotted (e.g., "Corrected sp Cond [uS/cm]" or "Vertical Position [m]").
        show_outliers (bool):
            Whether to display outliers in the boxplots.
        order (Optional[List[str]]):
            A list defining the order of the wells (well IDs) on the Y-axis.
            If a well ID is not listed here, it will be ignored.
            If not provided, the natural reading order of `filtered_data` is used.
        legend_x (float):
            X position where the legend is anchored.
        legend_y (float):
            Y position where the legend is anchored.
        legend_orientation (str):
            Legend orientation: 'vertical' or 'horizontal'.
        mirror_top_axis (bool):
            If True, adds a mirrored X-axis at the top.
        enable_minor_ticks (bool):
            If True, activates minor ticks on the X-axis every 1000 units.
        draw_vertical_line (bool):
            If True, draws a dotted vertical line at x=1452.9.

    Returns:
        (fig, ax): The Matplotlib Figure and Axes objects.
    """

    # Dictionary to assign a specific color to each filtering method
    method_colors = {
        'BIC': 'blue',
        'IC': 'red',
        'DGH': 'green'
    }

    # 1) Collect data in a list of tuples (well_id, method, values) 
    #    in the natural reading order of 'filtered_data'
    plot_data = []
    for well_id, methods_dict in filtered_data.items():
        for method, df in methods_dict.items():
            values = df[variable].values
            plot_data.append((well_id, method, values))

    # 2) If an 'order' list is provided, filter and reorder 'plot_data' 
    #    based on the position of the well_id in 'order'
    if order is not None:
        # Create a dictionary to store the index of each well ID in 'order'
        order_index = {well: i for i, well in enumerate(order)}

        # Keep only those tuples whose well_id is in 'order'
        plot_data = [item for item in plot_data if item[0] in order_index]

        # Sort by the index given in 'order'
        plot_data.sort(key=lambda x: order_index[x[0]])

    # 3) Build the final lists for plotting
    group_labels = []
    group_data = []
    method_list = []

    # Create labels, data arrays, and method entries for each tuple
    for well_id, method, values in plot_data:
        label = f"{well_id} - {method}"
        group_labels.append(label)
        group_data.append(values)
        method_list.append(method)

    # Create the figure and axes. 
    # The figure height depends on the number of labels.
    fig, ax = plt.subplots(figsize=(12, len(group_labels) * 0.5 + 2))

    # Positions on the Y-axis for each boxplot
    box_positions = np.arange(len(group_labels))

    # Decide how to display outliers: 'o' for showing, '' for hiding
    sym = 'o' if show_outliers else ''

    # Keep track of methods already added to the legend
    used_methods = set()

    # 4) Create boxplots grouped by method to ensure consistent colors
    unique_methods = sorted(set(method_list))
    for method in unique_methods:
        # Collect positions and data for this method
        method_positions = []
        method_data = []
        for i, (lbl, dat, meth) in enumerate(zip(group_labels, group_data, method_list)):
            if meth == method:
                method_positions.append(i)
                method_data.append(dat)

        # If there is data for this method, draw the boxplot
        if method_positions:
            color = method_colors.get(method, 'gray')
            bp = ax.boxplot(
                method_data,
                positions=method_positions,
                sym=sym,
                vert=False,
                widths=0.6,
                patch_artist=True
            )

            # Customize boxplot colors
            for box in bp['boxes']:
                box.set(color=color, facecolor=color, alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color=color)
            for cap in bp['caps']:
                cap.set(color=color)
            for median in bp['medians']:
                median.set(color='black')
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor=color, markersize=5, alpha=0.7)

            # Add legend entry for this method, but only once
            if method not in used_methods:
                ax.plot([], [], color=color, label=method)
                used_methods.add(method)

    # 5) Add annotations to the right side indicating 'n' and outliers
    for i, data in enumerate(group_data):
        n_points = len(data)
        n_outliers = calculate_outliers(data)
        annotation_text = f"n={n_points}, out={n_outliers}"
        x_max = ax.get_xlim()[1]
        ax.text(x_max * 1.05, i, annotation_text,
                verticalalignment='center', fontsize=10)

    # Configure the Y-axis with labels
    ax.set_yticks(box_positions)
    ax.set_yticklabels(group_labels)

    # Set axis labels and title
    ax.set_xlabel(variable)
    ax.set_ylabel("Well - Filtering Method")
    ax.set_title(f"Boxplots of {variable} by Well and Filtering Method",
                 pad=20, y=1.01)

    # Display major grid on the X-axis
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)

    # Optionally enable minor ticks every 1000 units on the X-axis
    if enable_minor_ticks:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2)

    # Optionally draw a vertical line at x=1452.9
    if draw_vertical_line:
        ax.axvline(x=1452.9, color='black', linestyle='--', label='x = 1452.9')

    # Optionally add a mirrored X-axis at the top
    if mirror_top_axis:
        ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True)

    # Adjust right margin to leave room for annotations
    plt.subplots_adjust(right=0.75)

    # Configure the legend
    legend_loc = 'upper right' if legend_x > 0.5 else 'upper left'
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        framealpha=0.8,
        title="Method"
    )

    # Final layout adjustments
    fig.tight_layout()

    # Return the figure and axes for further use if needed
    return fig, ax