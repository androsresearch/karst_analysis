# %%
import os
import sys
import warnings

root = os.path.abspath('.')  
sys.path.append(root)
# warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from typing import Tuple, Optional, Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 25)


# %%

# Load data
path_dgh = f'{root}/data/dgh_fwl_estimation.csv'

df_dgh = pd.read_csv(path_dgh)

df_dgh = df_dgh[['ID', 'breakpoint_1 (vp)', 'breakpoint_2 (vp)']]

df_dgh.head()

# %%

df_aw2d = pd.read_csv(f'{root}/data/rawdy/AW2D_YSI_20230815_rowdy.csv')

# Asignar el último punto de `ID` = AW2D_YSI_20230815 a breakpoint_2 (vp)
df_dgh.loc[df_dgh['ID'] == 'AW2D_YSI_20230815', 'breakpoint_2 (vp)'] = df_aw2d['Vertical Position [m]'].iloc[-1]

df_dgh.head(2)



# %%
def filter_rowdy_data(path_data, df_dgh, seawater_ec=55600, tolerance=0.1):
    """
    Filters rowdy.csv files based on df_dgh breakpoints into three zones:
    FWL, MZ and SZ. Also validates profiles based on maximum EC value.
    
    Parameters:
    -----------
    path_data : str
        Path to the data directory.
    df_dgh : pandas.DataFrame
        DataFrame with columns ID, breakpoint_1 (vp), breakpoint_2 (vp).
    seawater_ec : float, optional
        Reference seawater electrical conductivity in μS/cm. Default is 55600.
    tolerance : float, optional
        Tolerance margin for seawater EC validation (0.1 = 10%). Default is 0.1.
        
    Returns:
    --------
    tuple
        (results, excluded_profiles)
        - results: Dictionary with ID as key and dictionary with filtered arrays for FWL, MZ and SZ.
        - excluded_profiles: Dictionary with excluded profiles and their maximum EC values.
    """
    results = {}
    excluded_profiles = {}
    
    # Calculate acceptable EC range
    min_ec = seawater_ec * (1 - tolerance)
    
    for _, row in df_dgh.iterrows():
        id_val = row['ID']
        bp1_vp = row['breakpoint_1 (vp)']
        bp2_vp = row['breakpoint_2 (vp)']
        
        filename = f"{id_val}_rowdy.csv"
        filepath = os.path.join(path_data, filename)
        
        if os.path.exists(filepath):
            df_rowdy = pd.read_csv(filepath)
            
            df_rowdy.columns = [col.strip() for col in df_rowdy.columns]
            
            if 'Vertical Position [m]' in df_rowdy.columns and 'Corrected sp Cond [uS/cm]' in df_rowdy.columns:
                # Check if the profile meets the seawater EC criterion
                max_ec_value = df_rowdy['Corrected sp Cond [uS/cm]'].max()
                
                # Filter FWL zone (below breakpoint_1)
                fwl_mask = df_rowdy['Vertical Position [m]'] < bp1_vp
                fwl_data = df_rowdy[fwl_mask][['Vertical Position [m]', 'Corrected sp Cond [uS/cm]']].values
                
                # Filter MZ zone (between breakpoint_1 and breakpoint_2)
                mz_mask = (df_rowdy['Vertical Position [m]'] >= bp1_vp) & (df_rowdy['Vertical Position [m]'] <= bp2_vp)
                mz_data = df_rowdy[mz_mask][['Vertical Position [m]', 'Corrected sp Cond [uS/cm]']].values
                
                # Filter SZ zone (above breakpoint_2)
                sz_mask = df_rowdy['Vertical Position [m]'] > bp2_vp
                sz_data = df_rowdy[sz_mask][['Vertical Position [m]', 'Corrected sp Cond [uS/cm]']].values
                
                # Store data regardless of whether it meets EC criterion
                results[id_val] = {
                    'FWL': fwl_data,
                    'MZ': mz_data,
                    'SZ': sz_data
                }
                
                # Check if the profile should be excluded (doesn't meet seawater EC criterion)
                if max_ec_value < min_ec:
                    excluded_profiles[id_val] = {
                        'max_ec': max_ec_value,
                        'min_threshold': min_ec,
                        'reason': f"Profile doesn't reach salinity zone (max SEC {max_ec_value:.2f} μS/cm < threshold {min_ec:.2f} μS/cm)"
                    }
            else:
                print(f"File {filename} does not have expected columns.")
        else:
            print(f"File {filename} not found")
    
    return results, excluded_profiles

zones, excluded_wells = filter_rowdy_data(
                        f'{root}/data/rawdy', 
                        df_dgh=df_dgh,
                        seawater_ec=55600, # uS/cm
                        tolerance=0.1 # 10%
            )

print(excluded_wells)

# %%

def calculate_statistics(filter_results, excluded_profiles=None):
    """
    Calculate basic statistics for filtered data.
    
    Parameters:
    -----------
    filter_results : dict
        Dictionary with filtered results.
    excluded_profiles : dict, optional
        Dictionary with excluded profiles and reasons.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with basic statistics by zone and ID.
    """
    # List to store DataFrame rows
    stats_rows = []
    
    # Iterate over each ID and its data
    for id_val, zones_data in filter_results.items():
        is_excluded = excluded_profiles and id_val in excluded_profiles
        
        for zone, data in zones_data.items():
            # Special handling for excluded profiles
            if is_excluded and zone == 'SZ':
                # For excluded profiles, don't calculate SZ statistics
                stats_row = {
                    'ID': id_val,
                    'Zone': zone,
                    'mean': np.nan,
                    'std': np.nan,
                    'cv': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    '25%': np.nan,
                    '50%': np.nan,
                    '75%': np.nan,
                    'iqr': np.nan,
                    'count': 0,
                    'outliers': 0,
                    'thickness': 0,
                    'excluded': True,
                    'exclusion_reason': excluded_profiles[id_val]['reason'] + " - statistics not calculated for this zone."
                }
                stats_rows.append(stats_row)
                continue
                
            # Extract conductivity values (second column)
            cond_values = data[:, 1]
            
            if len(cond_values) > 0:
                # Calculate quartiles
                q1 = np.percentile(cond_values, 25)
                q2 = np.percentile(cond_values, 50)  # Median
                q3 = np.percentile(cond_values, 75)
                iqr_val = q3 - q1
                
                # Identify outliers using IQR method
                lower_bound = q1 - 1.5 * iqr_val
                upper_bound = q3 + 1.5 * iqr_val
                outliers = np.sum((cond_values < lower_bound) | (cond_values > upper_bound))
                
                # Calculate coefficient of variation (CV)
                mean_val = np.mean(cond_values)
                std_val = np.std(cond_values)
                cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
                
                # Calculate zone thickness (max depth - min depth)
                depth_values = data[:, 0]
                thickness = np.max(depth_values) - np.min(depth_values) if len(depth_values) > 0 else 0
                
                # Create row with statistics
                stats_row = {
                    'ID': id_val,
                    'Zone': zone,
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val,
                    'min': np.min(cond_values),
                    'max': np.max(cond_values),
                    'median': q2,
                    '25%': q1,
                    '50%': q2,
                    '75%': q3,
                    'iqr': iqr_val,
                    'count': len(cond_values),
                    'outliers': outliers,
                    'thickness': thickness,
                    'excluded': is_excluded
                }
                
                # Add note for MZ in excluded profiles
                if is_excluded and zone == 'MZ':
                    stats_row['exclusion_reason'] = "Note: Statistics calculated for incomplete profiles (missing salinity zone)."
                
                stats_rows.append(stats_row)
            else:
                # Create row for empty zones
                stats_row = {
                    'ID': id_val,
                    'Zone': zone,
                    'mean': np.nan,
                    'std': np.nan,
                    'cv': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    '25%': np.nan,
                    '50%': np.nan,
                    '75%': np.nan,
                    'iqr': np.nan,
                    'count': 0,
                    'outliers': 0,
                    'thickness': 0,
                    'excluded': is_excluded
                }
                
                if is_excluded:
                    if zone == 'SZ':
                        stats_row['exclusion_reason'] = excluded_profiles[id_val]['reason'] + " - statistics not calculated for this zone."
                    elif zone == 'MZ':
                        stats_row['exclusion_reason'] = "Note: Statistics calculated for incomplete profiles (missing salinity zone)."
                
                stats_rows.append(stats_row)
    
    # Create DataFrame with statistics
    stats_df = pd.DataFrame(stats_rows)
    
    return stats_df

# %%

df_stats_zones = calculate_statistics(
                    filter_results=zones,
                    excluded_profiles=excluded_wells
            )


df_stats_zones

# %%

# (Optional)

# df_stats_zones.to_csv(f'{root}/data/stats_zones.csv', index=False)


# %%

def generate_zone_boxplots(
    filtered_results: Dict[str, Dict[str, np.ndarray]],
    zones_to_show: Optional[List[str]] = None,
    show_outliers: bool = True,
    order: Optional[List[str]] = None,
    legend_x: float = 1.02,
    legend_y: float = 1.0,
    mirror_top_axis: bool = False,
    enable_minor_ticks: bool = False,
    draw_vertical_line: bool = False,
    vertical_line_value: float = None,
    figsize: Tuple[int, int] = None,
    excluded_profiles: Optional[Dict[str, Dict[str, str]]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates horizontal boxplots for specific conductivity by ID and zone.

    Args:
        filtered_results: Nested dictionary containing filtered data
            - First key: Well ID
            - Second key: Zone ('FWL', 'MZ', 'SZ')
        zones_to_show: List of zones to display (e.g. ['FWL', 'MZ', 'SZ'])
            If not provided, shows all available zones
        show_outliers: Whether to show outliers in boxplots
        order: List defining well ID order on Y axis
            IDs not in list will be ignored
            If not provided, uses natural order from filtered_results
        legend_x: X position to anchor legend
        legend_y: Y position to anchor legend  
        legend_orientation: Legend orientation - 'vertical' or 'horizontal'
        mirror_top_axis: If True, adds mirrored X axis on top
        enable_minor_ticks: If True, enables minor ticks every 1000 units
        draw_vertical_line: If True, draws vertical dotted line
        vertical_line_value: X value for vertical line (only if draw_vertical_line is True)
        figsize: Custom figure size (width, height)
        excluded_profiles: Dictionary of profiles to exclude from the plot

    Returns:
        Figure and Axes objects from Matplotlib
    """
    zone_colors = {
        'FWL': 'green',
        'MZ': 'blue', 
        'SZ': 'red'
    }

    variable = 'Corrected sp Cond [uS/cm]'

    # Collect data in list of tuples (id, zone, values)
    plot_data = []
    for id_val, zones_dict in filtered_results.items():
        # No longer skip excluded profiles completely
        # Instead, handle each zone individually
        
        for zone, data in zones_dict.items():
            # Skip SZ zone for excluded profiles
            if excluded_profiles and id_val in excluded_profiles and zone == 'SZ':
                continue
                
            if data.shape[0] > 0:
                values = data[:, 1]  # Column 1 contains conductivity values
                # Calculate thickness (last depth - first depth)
                thickness = data[-1, 0] - data[0, 0]
                plot_data.append((id_val, zone, values, thickness))

    # Filter by zones_to_show if provided
    if zones_to_show is not None:
        plot_data = [item for item in plot_data if item[1] in zones_to_show]

    # Filter and reorder by order list if provided
    if order is not None:
        order_index = {well: i for i, well in enumerate(order)}
        plot_data = [item for item in plot_data if item[0] in order_index]
        plot_data.sort(key=lambda x: order_index[x[0]])

    # Build final lists for plotting
    group_labels = []
    group_data = []
    zone_list = []
    thickness_list = []
    excluded_flags = []  # Para marcar los perfiles excluidos

    for id_val, zone, values, thickness in plot_data:
        label = f"{id_val} - {zone}"
        # Añadir un indicador visual para perfiles excluidos (excepto SZ que ya está excluido)
        if excluded_profiles and id_val in excluded_profiles:
            if zone in ['FWL', 'MZ']:
                label += " *"  # Marca con asterisco los perfiles excluidos
                excluded_flags.append(True)
            else:
                excluded_flags.append(False)
        else:
            excluded_flags.append(False)
            
        group_labels.append(label)
        group_data.append(values)
        zone_list.append(zone)
        thickness_list.append(thickness)

    # Create figure and axes
    if figsize is None:
        figsize = (12, len(group_labels) * 0.5 + 2)
    
    fig, ax = plt.subplots(figsize=figsize)

    box_positions = np.arange(len(group_labels))
    sym = 'o' if show_outliers else ''
    used_zones = set()

    # Create boxplots grouped by zone
    unique_zones = sorted(set(zone_list))
    for zone in unique_zones:
        zone_positions = []
        zone_data = []
        for i, (lbl, dat, z) in enumerate(zip(group_labels, group_data, zone_list)):
            if z == zone:
                zone_positions.append(i)
                zone_data.append(dat)

        if zone_positions:
            color = zone_colors.get(zone, 'gray')
            bp = ax.boxplot(
                zone_data,
                positions=zone_positions,
                sym=sym,
                vert=False,
                widths=0.6,
                patch_artist=True
            )

            # Style boxplots
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

            if zone not in used_zones:
                ax.plot([], [], color=color, label=zone)
                used_zones.add(zone)

    def _calculate_outliers(data):
        """Calculate number of outliers using IQR method"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = sum((data < lower_bound) | (data > upper_bound))
        return outliers

    # Add n, out and thickness annotations
    for i, (data, thickness, is_excluded) in enumerate(zip(group_data, thickness_list, excluded_flags)):
        n_points = len(data)
        n_outliers = _calculate_outliers(data)
        annotation_text = f"n={n_points}, out={n_outliers}, t={thickness:.3f}"
        if is_excluded:
            annotation_text += " (Incomplete profile)"
        x_max = ax.get_xlim()[1]
        ax.text(x_max * 1.05, i, annotation_text,
                verticalalignment='center', fontsize=10)

    # Configure axes and labels
    ax.set_yticks(box_positions)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel('Corrected sp Cond [uS/cm]')
    ax.set_ylabel("Well ID - Zone")
    ax.set_title("Boxplots of Specific Conductivity by Well ID and Zone",
                pad=20, y=1.01)

    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)

    if enable_minor_ticks:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
        ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2)

    if draw_vertical_line and vertical_line_value is not None:
        ax.axvline(x=vertical_line_value, color='black', linestyle='--', 
                   label=f'x = {vertical_line_value}')

    if mirror_top_axis:
        ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True)

    plt.subplots_adjust(right=0.75)

    legend_loc = 'upper right' if legend_x > 0.5 else 'upper left'
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        framealpha=0.8,
        title="Zone"
    )
    
    if excluded_profiles:
        excluded_ids = list(excluded_profiles.keys())
        excluded_text = "Excluded wells (SZ zone only):\n" + "\n".join(excluded_ids)
        excluded_reason = "Reason: " + next(iter(excluded_profiles.values()))['reason']
        excluded_text += f"\n{excluded_reason}"
        # Colocar la anotación en la esquina inferior izquierda
        plt.figtext(0.025, -0.01, excluded_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    fig.tight_layout()

    return fig, ax

# %%
# From bottom to top in the graph.

order_plot = [
    "LRS70D_YSI_20230822", 
    "BW9D_YSI_20230823", 
    "BW11D_YSI_20230823",
    "BW8D_YSI_20230823",
    "LRS81D_YSI_20230823",
    "BW1D_YSI_20230824",
    "LRS79D_YSI_20230827",
    "AW1D_YSI_20230826",
    "LRS90D_YSI_20230827",
    "AW2D_YSI_20230815",
    "AW5D_YSI_20230824",
    "BW2D_YSI_20230819",
    "LRS75D_YSI_20230819",
    "LRS89D_YSI_20230825",
    'LRS65D_YSI_20220812',
    "AW7D_YSI_20230814",
    "AW6D_YSI_20230815",
    "BW3D_YSI_20230818",
    "BW10D_YSI_20230825",
    "BW6D_YSI_20230826",
    "BW4D_YSI_20230816",
    "LRS69D_YSI_20230818",
    "LRS33D_YSI_20230822",
    "BW7D_YSI_20230826",
    "BW5D_YSI_20230822" 
]

# %%

fig, ax = generate_zone_boxplots(filtered_results=zones,
                            zones_to_show=['SZ', 'MZ', 'FWL'],
                            show_outliers=False,
                            order=order_plot,
                            # enable_minor_ticks=True
                            mirror_top_axis=True,
                            legend_x=0.01,
                            legend_y=1.0,
                            excluded_profiles=excluded_wells
                            )

fig.show()