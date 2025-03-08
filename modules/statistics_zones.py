import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def split_by_points(A: float, B: float, df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y'):
    """
    Splits the input DataFrame into three segments based on the x-values using two threshold points A and B.
    
    The segments are defined as:
        - Segment 1: x < A
        - Segment 2: A ≤ x ≤ B
        - Segment 3: x > B

    Parameters:
    -----------
    A : float
        The lower threshold for segmentation.
    B : float
        The upper threshold for segmentation.
    df : pd.DataFrame
        The input DataFrame containing the data.
    x_col : str, optional
        The name of the column to be used for segmentation (default is 'x').
    y_col : str, optional
        The name of the column for associated values (default is 'y').

    Returns:
    --------
    tuple of pd.DataFrame
        A tuple containing three DataFrames corresponding to the segments:
            - First DataFrame: rows where x < A
            - Second DataFrame: rows where A ≤ x ≤ B
            - Third DataFrame: rows where x > B

    Raises:
    -------
    KeyError:
        If the specified x_col or y_col do not exist in the DataFrame.
    ValueError:
        If A is greater than B.
    TypeError:
        If the DataFrame is not a pandas DataFrame.
    """
    # Error handling: Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The provided data is not a pandas DataFrame.")
    
    # Error handling: Check if required columns exist
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Error handling: Check if A is less than or equal to B
    if A > B:
        raise ValueError("Threshold A should be less than or equal to threshold B.")

    # Using numpy for efficient boolean indexing
    x_values = df[x_col].values

    mask1 = x_values < A
    mask2 = (x_values >= A) & (x_values <= B)
    mask3 = x_values > B

    segment1 = df[mask1].copy()
    segment2 = df[mask2].copy()
    segment3 = df[mask3].copy()

    return segment1, segment2, segment3

def compute_segment_statistics(seg1: pd.DataFrame, 
                               seg2: pd.DataFrame, 
                               seg3: pd.DataFrame, 
                               column: str) -> pd.DataFrame:
    """
    Computes descriptive statistics for a specified column in each of the three DataFrame segments.
    
    The computed metrics for each segment include:
        - mean: Average of the values.
        - std: Standard deviation of the values.
        - cv: Coefficient of variation (std / mean).
        - min: Minimum value.
        - max: Maximum value.
        - median: Median value.
        - 25%: 25th percentile.
        - 50%: 50th percentile (same as median).
        - 75%: 75th percentile.
        - iqr: Interquartile range (75th percentile - 25th percentile).

    Parameters:
    -----------
    seg1 : pd.DataFrame
        First segment DataFrame.
    seg2 : pd.DataFrame
        Second segment DataFrame.
    seg3 : pd.DataFrame
        Third segment DataFrame.
    column : str
        The name of the column for which to compute the statistics.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to a segment (Segment 1, Segment 2, Segment 3)
        and each column contains one of the computed metrics.
    
    Raises:
    -------
    KeyError:
        If the specified column does not exist in any of the segments.
    TypeError:
        If the provided segments are not pandas DataFrames.
    """
    
    # Helper function to compute metrics for a given series.
    def compute_metrics(series: pd.Series) -> dict:
        if series.empty:
            # If the segment is empty, return NaN for all metrics.
            return {
                'mean': np.nan,
                'std': np.nan,
                'cv': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
                '25%': np.nan,
                '50%': np.nan,
                '75%': np.nan,
                'iqr': np.nan
            }
        mean_val = series.mean()
        std_val = series.std()
        # Avoid division by zero; if mean is 0, cv will be NaN.
        cv_val = std_val / mean_val if mean_val != 0 else np.nan
        q25 = series.quantile(0.25)
        q50 = series.quantile(0.50)  # same as median
        q75 = series.quantile(0.75)
        iqr_val = q75 - q25

        return {
            'mean': mean_val,
            'std': std_val,
            'cv': cv_val,
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            '25%': q25,
            '50%': q50,
            '75%': q75,
            'iqr': iqr_val
        }
    
    # Validate that each segment is a DataFrame and contains the specified column.
    for idx, seg in enumerate([seg1, seg2, seg3], start=1):
        if not isinstance(seg, pd.DataFrame):
            raise TypeError(f"Segment {idx} is not a pandas DataFrame.")
        if column not in seg.columns:
            raise KeyError(f"Column '{column}' not found in segment {idx}.")

    # Compute metrics for each segment.
    stats_seg1 = compute_metrics(seg1[column])
    stats_seg2 = compute_metrics(seg2[column])
    stats_seg3 = compute_metrics(seg3[column])
    
    # Combine the results into a DataFrame.
    results = pd.DataFrame({
        'Freshwater zone': stats_seg1,
        'Mixing zone': stats_seg2,
        'Saltwater zone': stats_seg3
    }).T  # Transpose to have segments as rows
    
    return results

def plot_boxplot_segments(seg1: pd.DataFrame, 
                          seg2: pd.DataFrame, 
                          seg3: pd.DataFrame, 
                          variable: str, 
                          segment_names: list = None, 
                          show_outliers: bool = True,
                          title: str = None) -> None:
    """
    Genera un gráfico de boxplots horizontales para la variable numérica indicada en tres segmentos,
    junto con anotaciones que muestran el número total de puntos y la cantidad de outliers (calculados
    mediante el método de Tukey) para cada segmento.
    
    La figura resultante consta de:
      - La primera columna muestra el boxplot de la distribución de la variable seleccionada para cada segmento.
      - La segunda columna muestra, para cada segmento, el número total de puntos y la cantidad de outliers.
      - El eje y muestra los nombres de los segmentos (se pueden personalizar mediante el parámetro `segment_names`).
      - El eje x del boxplot muestra los valores de la variable seleccionada.
      - El parámetro `show_outliers` controla si se muestran o no los outliers en el boxplot.
      - Se puede agregar un título personalizado mediante el parámetro `title`.
    
    Parámetros:
    -----------
    seg1 : pd.DataFrame
        DataFrame correspondiente al primer segmento.
    seg2 : pd.DataFrame
        DataFrame correspondiente al segundo segmento.
    seg3 : pd.DataFrame
        DataFrame correspondiente al tercer segmento.
    variable : str
        Nombre de la columna numérica a graficar.
    segment_names : list of str, optional
        Lista con los nombres de los segmentos. Por defecto es ['Segment 1', 'Segment 2', 'Segment 3'].
    show_outliers : bool, optional
        Si es True, se muestran los outliers en el boxplot; de lo contrario se ocultan.
    title : str, optional
        Título del gráfico. Si no se proporciona, se genera un título por defecto.
    
    Raises:
    -------
    KeyError:
        Si la variable especificada no se encuentra en alguno de los segmentos.
    TypeError:
        Si alguno de los segmentos no es un DataFrame de pandas.
    """
    if segment_names is None:
        segment_names = ['Segment 1', 'Segment 2', 'Segment 3']

    segments = [seg1, seg2, seg3]
    for idx, seg in enumerate(segments, start=1):
        if not isinstance(seg, pd.DataFrame):
            raise TypeError(f"El segmento {idx} no es un DataFrame de pandas.")
        if variable not in seg.columns:
            raise KeyError(f"La variable '{variable}' no se encontró en el segmento {idx}.")
    
    data_series = [seg[variable] for seg in segments]
    
    if title is None:
        title = f"Boxplot de '{variable}' por segmento"
    
    fig = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True,
        column_widths=[0.75, 0.25],
        horizontal_spacing=0.05
    )
    
    for i, series in enumerate(data_series):
        fig.add_trace(
            go.Box(
                x=series,
                name=segment_names[i],
                orientation='h',
                boxpoints="outliers" if show_outliers else False,
                marker=dict(opacity=0.7),
                line=dict(width=1)
            ),
            row=1, col=1
        )
    
    annotation_texts = []
    for series in data_series:
        n_points = series.shape[0]
        if n_points > 0:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = series[(series < lower_bound) | (series > upper_bound)].shape[0]
        else:
            outlier_count = 0
        annotation_texts.append(f"n: {n_points}<br>Outliers: {outlier_count}")
    
    fig.add_trace(
        go.Scatter(
            x=[0.5] * len(segment_names), 
            y=segment_names,
            mode="text",
            text=annotation_texts,
            textfont=dict(size=12)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
        margin=dict(l=50, r=50, t=70, b=50),
        showlegend=False
    )
    
    fig.update_xaxes(title_text=variable, row=1, col=1)
    fig.update_yaxes(title_text="Segments", row=1, col=1)
    
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.show()
