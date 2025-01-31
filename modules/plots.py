import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple
from . import analysis
from math import ceil
import os
import glob
import pandas as pd


def plot_data(
    x_values: List[float],
    y_values: List[float],
    plot_mode: str = 'lines',
    trace_names: Optional[List[str]] = None,
    secondary_x: Optional[List[float]] = None,
    secondary_y: Optional[List[float]] = None,
    title: str = "",
    x_axis_label: str = "Vertical Position [m]",
    y_axis_label: str = "Corrected sp Cond [µS/cm]",
    y2_axis_label: str = "Secondary Y-Axis",
    use_secondary_axis: bool = False,
    enable_error_x: bool = False,
    enable_error_y: bool = False
) -> None:
    """
    Plot data using Plotly with customizable aesthetics and optional error bars.

    Parameters:
    x_values (List[float]): Data for the X-axis of the primary trace.
    y_values (List[float]): Data for the Y-axis of the primary trace.
    plot_mode (str): Plot mode ('lines', 'markers', 'lines+markers'). Defaults to 'lines'.
    trace_names (Optional[List[str]]): Names for the traces. Defaults to ['Primary Trace', 'Secondary Trace'].
    secondary_x (Optional[List[float]]): Data for the X-axis of the secondary trace.
    secondary_y (Optional[List[float]]): Data for the Y-axis of the secondary trace.
    title (str): Plot title.
    x_axis_label (str): Label for the X-axis.
    y_axis_label (str): Label for the primary Y-axis.
    y2_axis_label (str): Label for the secondary Y-axis (if use_secondary_axis=True).
    use_secondary_axis (bool): Enable a second Y-axis with its own scale.
    enable_error_x (bool): Add error bars of +/- 0.001 on the X-axis if True.
    enable_error_y (bool): Add error bars of +/- 0.5% on the Y-axis if True.

    Returns:
    None
    """
    trace_names = trace_names or ['Primary Trace', 'Secondary Trace']

    error_x = {'type': 'constant', 'value': 0.001} if enable_error_x else None
    error_y = {'type': 'percent', 'value': 0.5} if enable_error_y else None

    fig = (
        make_subplots(specs=[[{"secondary_y": True}]])
        if use_secondary_axis else go.Figure()
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode=plot_mode,
            name=f"{trace_names[0]} (n = {len(x_values)})",
            error_x=error_x,
            error_y=error_y
        ),
        secondary_y=False if use_secondary_axis else None
    )

    if secondary_x is not None and secondary_y is not None:
        fig.add_trace(
            go.Scatter(
                x=secondary_x,
                y=secondary_y,
                mode=plot_mode,
                name=f"{trace_names[1]} (n = {len(secondary_x)})",
                error_x=error_x,
                error_y=error_y
            ),
            secondary_y=True if use_secondary_axis else None
        )

    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_axis_label,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='Gray'
    )

    fig.update_yaxes(
        title_text=y_axis_label,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='Gray',
        secondary_y=False if use_secondary_axis else None
    )

    if use_secondary_axis:
        fig.update_yaxes(
            title_text=y2_axis_label,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='Gray',
            secondary_y=True
        )

    fig.show()

def plot_histogram(
    data: Any,
    value_column: str = 'Value',
    weight_column: Optional[str] = 'Weight',
    num_bins: int = 20,
    title: str = "",
    x_axis_title: str = "Values",
    y_axis_title: str = "Frequency",
    bar_color: str = "blue",
    line_color: str = "black",
    line_width: int = 1,
    bar_gap: float = 0.1,
    template: str = "plotly_white"
) -> go.Figure:
    """
    Generate an interactive histogram using Plotly.

    Parameters:
    data (Any): DataFrame containing the data.
    value_column (str): Column name for values to bin.
    weight_column (Optional[str]): Column name for weights (optional).
    num_bins (int): Number of bins for the histogram. Defaults to 20.
    title (str): Plot title.
    x_axis_title (str): Label for the X-axis.
    y_axis_title (str): Label for the Y-axis.
    bar_color (str): Color of the bars.
    line_color (str): Color of the bar borders.
    line_width (int): Width of the bar borders.
    bar_gap (float): Spacing between bars.
    template (str): Plotly template. Defaults to "plotly_white".

    Returns:
    go.Figure: The generated histogram figure.
    """
    hist, bin_edges = np.histogram(
        data[value_column],
        bins=num_bins,
        weights=data[weight_column] if weight_column else None
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        text=hist,
        textposition='auto',
        marker=dict(color=bar_color, line=dict(color=line_color, width=line_width))
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        bargap=bar_gap,
        template=template
    )

    return fig

def plot_segments(segments_info, metrics):
    """
    Plot each segment with its corresponding data and fitted model in subplots.

    Parameters:
    - segments_info (dict): Dictionary returned by `extract_segments`.
    - metrics (list): A list of dictionaries containing metrics for each segment.

    """
    segments = segments_info["segments"]
    n_segments = len(segments)

    # Determine subplot grid layout
    n_rows = ceil(np.sqrt(n_segments))
    n_cols = ceil(n_segments / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), squeeze=False)
    axes = axes.ravel()

    for i, segment in enumerate(segments):
        ax = axes[i]

        # Data for the current segment
        x = segment["data_x"]
        y = segment["data_y"]
        y_fit = segment["fitted_model"]["fitted_y"]

        # Plot data points and fitted line
        ax.scatter(x, y, label="Data", alpha=0.7)
        ax.plot(x, y_fit, color="red", label="Fit")

        # Add metrics as text inside the subplot
        metric = next((m for m in metrics if m["Segment"] == segment["segment"]), None)
        if metric:
            metric_text = "\n".join([
                f"R²: {metric['R^2']:.3f}",
                f"RMS%: {metric['RMS%']:.3f}",
                f"RMS% (min-max): {metric['RMS% (min-max)']:.3f}"
            ])
            ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Labels and title
        ax.set_title(f"Segment {segment['segment']}: {len(x)} points")
        ax.set_xlabel("Vertical Position [m]")
        ax.set_ylabel("Corrected sp Cond [uS/cm]")
        ax.legend()

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def interactive_segmented_regression(
    x: np.ndarray,
    y: np.ndarray,
    df: Any
) -> None:
    """
    Generate an interactive plot for segmented regression with 0 to 10 breakpoints.

    Parameters:
    -----------
    x (np.ndarray): Independent variable values.
    y (np.ndarray): Dependent variable values.
    df (Any): DataFrame with columns:
        - 'n_breakpoints': Number of breakpoints (0 to 10).
        - 'estimates': Dictionary with keys and metrics of interest.

    Returns:
    --------
    None
    """
    unique_breakpoints = sorted(df['n_breakpoints'].unique())
    
    def extract_estimate(param: Any) -> float:
        return param.get('estimate', 0.0) if isinstance(param, dict) else param

    @interact(n_breakpoints=IntSlider(min=0, max=10, step=1, value=0))
    def update_plot(n_breakpoints: int = 0) -> None:
        row = df[df['n_breakpoints'] == n_breakpoints]
        if row.empty:
            print(f"No parameters for {n_breakpoints} breakpoints.")
            return

        row = row.iloc[0]
        estimates = row['estimates']

        c = extract_estimate(estimates['const'])
        alpha1 = extract_estimate(estimates['alpha1'])

        betas = [extract_estimate(estimates[f'beta{i}']) for i in range(1, n_breakpoints + 1)]
        breakpoints = [extract_estimate(estimates[f'breakpoint{i}']) for i in range(1, n_breakpoints + 1)]

        x_sorted = np.sort(np.array(x))
        y_hat = []
        for xx in x_sorted:
            val = c + alpha1 * xx
            for b, bp in zip(betas, breakpoints):
                if xx > bp:
                    val += b * (xx - bp)
            y_hat.append(val)

        y_pred = []
        for xx in x:
            val = c + alpha1 * xx
            for b, bp in zip(betas, breakpoints):
                if xx > bp:
                    val += b * (xx - bp)
            y_pred.append(val)

        p = 2 + 2 * n_breakpoints
        RSS, TSS, R2, R2_ajus = analysis.get_global_metrics(np.array(y), np.array(y_pred), p)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', alpha=0.6, label='Datos Reales')
        plt.plot(x_sorted, y_hat, color='darkorange', lw=3, label='Ajuste Segmentado')

        for bp in breakpoints:
            val_bp = c + alpha1 * bp
            for b, bp_j in zip(betas, breakpoints):
                if bp > bp_j:
                    val_bp += b * (bp - bp_j)
            plt.scatter(bp, val_bp, color='limegreen', s=100, edgecolors='k', zorder=5)

        plt.xlabel('Vertical Position [m]')
        plt.ylabel('Corrected sp Cond [uS/cm]')
        plt.title(f'Segmented Regression with {n_breakpoints} Breakpoint(s): ({len(x)}) points')

        plt.text(
            0.05, 0.95,
            f"RSS: {RSS:.2f}\nTSS: {TSS:.2f}\n$R^2$: {R2:.4f}\n$R^2$ Ajustado: {R2_ajus:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        plt.legend()
        plt.show()

def plot_sp_cond_boxplots(
    input_path: str,
    filenames: list = None,
    show_outliers: bool = True
):
    """
    Genera y muestra una figura con boxplots horizontales de la columna
    'Corrected sp Cond [uS/cm]' para cada archivo CSV especificado.

    Parameters
    ----------
    input_path : str
        Ruta del directorio que contiene los archivos CSV.
    filenames : list, optional
        Lista de nombres de archivos CSV a incluir en la figura.
        Si no se proporciona, se usarán todos los archivos .csv en `input_path`.
    show_outliers : bool, optional
        Si es True, los outliers se muestran en el boxplot. Si es False, se ocultan.
    """

    if filenames is None:
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        csv_files = [os.path.basename(f) for f in csv_files]
    else:
        csv_files = filenames

    if not csv_files:
        print("No se encontraron archivos CSV en la ruta especificada.")
        return

    data_list = []
    labels_list = []

    for file in csv_files:
        file_path = os.path.join(input_path, file)

        if not os.path.exists(file_path):
            print(f"El archivo '{file}' no se encontró en '{input_path}'. Se omite.")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"No fue posible leer '{file}'. Error: {str(e)}")
            continue

        if 'Corrected sp Cond [uS/cm]' not in df.columns:
            print(f"El archivo '{file}' no contiene la columna 'Corrected sp Cond [uS/cm]'. Se omite.")
            continue

        sp_cond_values = df['Corrected sp Cond [uS/cm]'].dropna()

        if sp_cond_values.empty:
            print(f"No hay datos en la columna 'Corrected sp Cond [uS/cm]' en '{file}'. Se omite.")
            continue
        
        data_list.append(sp_cond_values)

        label_clean = file.replace("_filter.csv", "")
        labels_list.append(label_clean)

    # Verificar que se haya podido leer al menos un archivo con datos válidos
    if not data_list:
        print("No se generarán boxplots porque no se encontraron datos válidos.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(
        data_list,
        labels=labels_list,
        vert=False,
        patch_artist=True, 
        showfliers=show_outliers  
    )

    ax.set_xlabel("Corrected sp Cond [uS/cm]")
    ax.set_ylabel("Freshwater Lens")
    ax.set_title("Boxplots 'Corrected sp Cond [uS/cm]'")

    for box in ax.artists:
        box.set_facecolor("#87CEEB")  

    plt.tight_layout()
    plt.show()

    return fig