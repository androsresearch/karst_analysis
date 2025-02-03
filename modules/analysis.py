import numpy as np
import pandas as pd
import piecewise_regression as pw
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Union, Dict, Any
from collections import Counter
from piecewise_regression.main import Fit
import json
import glob
import os

def elbow_max_distance(metric: Union[pd.Series, np.ndarray]) -> int:
    """
    Finds the elbow (optimal point) in a metric sequence using the maximum distance method.

    Parameters:
        metric (Union[pd.Series, np.ndarray]): Sequence of metric values evaluated for different configurations.

    Returns:
        int: Index of the elbow (optimal point).
    """
    # Ensure the metric is a numpy array
    if isinstance(metric, pd.Series):
        metric = metric.values

    if not isinstance(metric, np.ndarray):
        raise ValueError("Input 'metric' must be a pandas.Series or numpy.ndarray.")

    if len(metric) < 2:
        raise ValueError("Input 'metric' must have at least two elements to calculate an elbow.")

    # Coordinates of the start and end points of the sequence
    start_idx, end_idx = 0, len(metric) - 1
    start_value, end_value = metric[start_idx], metric[end_idx]

    # Vector representation of the line connecting start and end points
    line_vec = np.array([end_idx - start_idx, end_value - start_value])
    line_vec = line_vec / np.linalg.norm(line_vec)  # Normalize the vector

    # Compute the perpendicular distances of each point to the line
    distances = []
    for i in range(len(metric)):
        point_vec = np.array([i - start_idx, metric[i] - start_value])
        proj_length = np.dot(point_vec, line_vec)
        proj_vec = proj_length * line_vec
        distance_vec = point_vec - proj_vec
        distances.append(np.linalg.norm(distance_vec))

    # Index of the point with the maximum distance to the line
    elbow_index = int(np.argmax(distances))

    return elbow_index

def best_n_breakpoints(x, y, max_breakpoints: int = 10, n_trials: int = 3) -> Dict[str, Any]:
    """
    Finds the optimal number of breakpoints based on BIC and an additional metric (e.g., RSS) 
    and stores the results in a dictionary.

    Parameters:
        x (array-like): Input data (X-axis).
        y (array-like): Output data (Y-axis).
        max_breakpoints (int): Maximum number of breakpoints allowed.
        n_trials (int): Number of trials to ensure stability.

    Returns:
        dict: Dictionary where each key corresponds to a trial, containing:
              {
                'df': DataFrame with the trial's model summaries,
                'best_n_breakpoint_bic': Optimal number of breakpoints based on the elbow method for BIC,
                'min_bic_n_breakpoint': Number of breakpoints with the lowest BIC value,
                'best_n_breakpoint_rss': Optimal number of breakpoints based on the elbow method for RSS
              }
    """
    results = {}

    for i in range(n_trials):
        # Generate model and retrieve DataFrame with model summaries
        ms = pw.ModelSelection(x, y, max_breakpoints)
        ms_df = pd.DataFrame(ms.model_summaries)

        # Extract BIC and RSS values
        y_bic = ms_df['bic']
        y_rss = ms_df['rss']

        # Determine optimal breakpoints using the elbow method
        best_n_breakpoint_bic = elbow_max_distance(y_bic)
        best_n_breakpoint_rss = elbow_max_distance(y_rss)

        # Find the number of breakpoints corresponding to the minimum BIC
        min_bic_index = y_bic.idxmin()
        min_bic_n_breakpoint = ms_df.loc[min_bic_index, 'n_breakpoints']

        # Store results in the dictionary
        results[f'trial_{i+1}'] = {
            'df': ms_df,
            'best_n_breakpoint_bic': best_n_breakpoint_bic,
            'min_bic_n_breakpoint': min_bic_n_breakpoint,
            'best_n_breakpoint_rss': best_n_breakpoint_rss
        }

    return results

def select_best_trial(file_path, key="best_n_breakpoint_bic"):
    """
    Selecciona el mejor trial basado en un criterio definido.
    
    :param file_path: Ruta al archivo JSON con los resultados.
    :param key: Clave para seleccionar el criterio ("best_n_breakpoint_bic" o "best_n_breakpoint_rss").
    :return: Tupla con el nombre del mejor trial, los datos y el valor promedio.
    """
    # Leer el archivo JSON
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Contar las ocurrencias de los valores en la clave seleccionada
    counts = Counter([trial[key] for trial in data.values()])
    most_common_value = counts.most_common(1)[0][0]

    # Filtrar los trials que tienen ese número de puntos de quiebre
    filtered_trials = {
        trial_name: trial for trial_name, trial in data.items() if trial[key] == most_common_value
    }

    # Calcular el promedio de BIC o RSS para los trials filtrados
    best_trial_name = None
    lowest_average = float("inf")
    for trial_name, trial in filtered_trials.items():
        # Corregir el acceso a las claves
        metric_key = "bic" if key == "best_n_breakpoint_bic" else "rss"
        values = list(trial["df"][metric_key].values())
        average_value = sum(values) / len(values)
        if average_value < lowest_average:
            lowest_average = average_value
            best_trial_name = trial_name
    
    # Retornar el mejor trial
    return best_trial_name, data[best_trial_name], lowest_average

def calculate_density(x: np.ndarray, y: np.ndarray, bin_width: float = 1.0) -> pd.DataFrame:
    """
    Calculates the point density based on the `x` (meters) and `y` values.

    Parameters
    ----------
    x : np.ndarray
        X-axis coordinates.
    y : np.ndarray
        Y-axis coordinates.
    bin_width : float, optional
        Bin width in meters for grouping `x` values (default is 1.0).

    Returns
    -------
    density : pd.DataFrame
        DataFrame with the frequency of points per unit of `x`.
    """
    # Create a DataFrame with x and y values
    df = pd.DataFrame({'x': x, 'y': y})

    # Discretize x into bins of size bin_width
    df['x_bin'] = (df['x'] // bin_width) * bin_width

    # Group by x bins and count the frequency of y
    density = df.groupby('x_bin').size().reset_index(name='frequency')

    return density

def extract_breakpoints(model):
    """
    Extracts the x and y positions of the breakpoints and their confidence intervals from a Fit model.

    Parameters:
    - model: piecewise_regression.main.Fit
        A fitted piecewise regression model of type Fit.

    Returns:
    - pd.DataFrame
        A DataFrame containing the x positions, y positions, and confidence intervals for the breakpoints.
    """
    if not isinstance(model, Fit):
        raise TypeError("The provided model must be an instance of piecewise_regression.main.Fit")

    if not model.best_muggeo:
        raise ValueError("The model has not converged or does not contain valid breakpoints.")

    # Extract estimates from the best Muggeo fit
    estimates = model.best_muggeo.best_fit.estimates

    # Prepare lists for data
    breakpoints_x = []
    breakpoints_y = []
    conf_intervals = []

    # Loop through breakpoints and extract data
    for i in range(1, model.best_muggeo.n_breakpoints + 1):
        bp_key = f"breakpoint{i}"

        # Get x position and confidence interval
        bp_x = estimates[bp_key]["estimate"]
        bp_conf = estimates[bp_key]["confidence_interval"]

        # Calculate y position using the model's predict function
        bp_y = model.predict(np.array([bp_x]))[0]

        # Append to lists
        breakpoints_x.append(bp_x)
        breakpoints_y.append(bp_y)
        conf_intervals.append(bp_conf)

    # Create a DataFrame
    df = pd.DataFrame({
        "Breakpoint X Position": breakpoints_x,
        "Breakpoint Y Position": breakpoints_y,
        "Confidence Interval (X)": conf_intervals
    })

    return df

def get_global_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> tuple:
    """
    Calcula RSS, TSS, R^2 y R^2 ajustado.

    Parameters:
    ----------
    y_true : np.ndarray
        Valores reales de la variable dependiente.
    y_pred : np.ndarray
        Valores predichos por el modelo.
    p : int
        Número de parámetros estimados en el modelo.

    Returns:
    -------
    RSS : float
        Suma de cuadrados de los residuos.
    TSS : float
        Suma de cuadrados total (respecto a la media).
    R2 : float
        Coeficiente de determinación.
    R2_ajustado : float
        Coeficiente de determinación ajustado.
    """
    # RSS
    RSS = np.sum((y_true - y_pred) ** 2)
    # TSS
    TSS = np.sum((y_true - np.mean(y_true)) ** 2)
    # R^2
    R2 = 1 - (RSS / TSS)
    # R^2 ajustado
    n = len(y_true)
    R2_ajustado = 1 - (1 - R2) * (n - 1) / (n - p - 1) if (n - p - 1) != 0 else np.nan

    return RSS, TSS, R2, R2_ajustado

def calculate_metrics_per_segment(fit_model):
    """
    Calcula \(R^2\) y RMS porcentual para cada segmento ajustado.

    :param fit_model: Modelo ajustado de `Fit`.
    :type fit_model: piecewise_regression.main.Fit

    :return: Lista de diccionarios con métricas por segmento.
    """
    if not fit_model.best_muggeo:
        raise ValueError("El modelo no está ajustado correctamente.")

    metrics = []
    xx = np.array(fit_model.xx)
    yy = np.array(fit_model.yy)
    breakpoints = [min(xx)] + list(fit_model.best_muggeo.best_fit.next_breakpoints) + [max(xx)]

    for i in range(len(breakpoints) - 1):
        mask = (xx >= breakpoints[i]) & (xx < breakpoints[i + 1])
        xx_segment = xx[mask]
        yy_segment = yy[mask]
        yy_predicted = fit_model.predict(xx_segment)

        # Calcular R^2
        rss = np.sum((yy_segment - yy_predicted) ** 2)
        tss = np.sum((yy_segment - np.mean(yy_segment)) ** 2)
        r_squared = 1 - rss / tss

        rms = np.sqrt(mean_squared_error(yy_segment, yy_predicted))
        rms_percent_min_max = (rms / (np.max(yy_segment) - np.min(yy_segment))) * 100 if np.max(yy_segment) != np.min(yy_segment) else 0

        # Calcular RMS porcentual
        rms_percent = np.sqrt(np.mean(((yy_segment - yy_predicted) / yy_segment) ** 2)) * 100

        metrics.append({"Segment": i + 1, "R^2": r_squared, "RMS%": rms_percent, "RMS% (min-max)": rms_percent_min_max})

    return metrics

# # # # analysys for a single segment # # # #
def segment_data(x:np.array, y:np.array, df:dict, num_breakpoints:int):

    """
    Segmenta los datos en x e y a partir de los breakpoints definidos en el JSON.

    Parámetros:
    - x: numpy array con los valores de x.
    - y: numpy array con los valores de y.
    - df: Diccionario que contiene los datos de breakpoints del JSON.
    - num_breakpoints: Número de breakpoints a considerar para segmentar los datos.

    Retorna:
    - Un diccionario con las claves como índices de segmento y valores como listas de pares (x, y) correspondientes.
    """
    # Extraer los breakpoints para el número de breakpoints dado
    key = str(num_breakpoints)
    if key not in df["estimates"]:
        raise ValueError(f"El número de breakpoints {num_breakpoints} no está disponible en el JSON.")

    breakpoints = []
    for i in range(1, num_breakpoints + 1):
        breakpoint_key = f"breakpoint{i}"
        if breakpoint_key in df["estimates"][key]:
            breakpoints.append(df["estimates"][key][breakpoint_key]["estimate"])
        else:
            raise ValueError(f"El breakpoint {i} no está disponible en los datos para {key} breakpoints.")

    # Asegurarse de que los breakpoints estén ordenados
    breakpoints = sorted(breakpoints)

    # Segmentar los datos
    segments = {}
    start_idx = 0

    for i, bp in enumerate(breakpoints):
        end_idx = np.searchsorted(x, bp, side="right")
        segments[str(i + 1)] = [x[start_idx:end_idx], y[start_idx:end_idx]]
        start_idx = end_idx

    # Último segmento
    segments[str(len(breakpoints) + 1)] = [x[start_idx:], y[start_idx:]]

    return segments

def fit_linear_models(segments):
    """
    Ajusta modelos lineales a cada segmento de datos y calcula métricas de evaluación.

    Parámetros:
    - segments: Diccionario con segmentos de datos, donde cada clave corresponde a un segmento
      y los valores son listas [x_segment, y_segment].

    Retorna:
    - Un diccionario con las claves como índices de segmento y valores como:
      - `model`: Modelo ajustado (sklearn.linear_model.LinearRegression).
      - `RMS`: Root Mean Square Error (Error cuadrático medio).
      - `RMS_percent`: RMS relativo al rango de valores de y.
      - `RMS_percent_meas`: RMS porcentual normalizado respecto a cada punto medido.
      - `R^2`: Coeficiente de determinación.
    """
    results = {}

    for segment_key, (x_segment, y_segment) in segments.items():
        if len(x_segment) == 0 or len(y_segment) == 0:
            results[segment_key] = {"error": "Segmento vacío o insuficiente para ajuste"}
            continue

        # Reshape x_segment para sklearn
        x_segment = np.array(x_segment).reshape(-1, 1)
        y_segment = np.array(y_segment)

        # Ajustar el modelo
        model = LinearRegression()
        model.fit(x_segment, y_segment)

        # Predicciones
        y_pred = model.predict(x_segment)

        # Calcular métricas
        rms = np.sqrt(mean_squared_error(y_segment, y_pred))
        rms_percent = (rms / (np.max(y_segment) - np.min(y_segment))) * 100 if np.max(y_segment) != np.min(y_segment) else 0

        # RMS porcentual basado en valores medidos
        if np.any(y_segment == 0):  # Evitar división por cero
            rms_percent_meas = "Indefinido (valores de y_segment contienen ceros)"
        else:
            rms_percent_meas = np.sqrt(
                np.mean(((y_pred - y_segment) / y_segment) ** 2)
            ) * 100

        r2 = r2_score(y_segment, y_pred)

        # Guardar resultados
        results[segment_key] = {
            "model": model,
            "RMS": rms,
            "RMS%_min_max": rms_percent,
            "RMS%": rms_percent_meas,
            "R^2": r2
        }

    return results

# # # #  Freshwater profiles analysis # # # #

def statistics_csv_files(
    input_folder: str,
    output_folder: str,
    name_folder: str,
    target_column: str = "Corrected sp Cond [uS/cm]"
) -> None:
    """
    Reads multiple CSV files from the given input folder, computes descriptive statistics
    for a specific numerical column, and saves the results in a CSV file inside the output folder.
    
    Args:
        input_folder (str): Path to the folder containing the input CSV files.
        output_folder (str): Path to the folder where the result CSV file will be saved.
        target_column (str, optional): Name of the column for which statistics will be calculated.
                                       Defaults to 'Corrected sp Cond [uS/cm]'.

    The function expects each CSV to have at least:
        - "Vertical Position [m]" (vertical position in meters)
        - "Corrected sp Cond [uS/cm]" (specific conductance in µS/cm), or 
          a column indicated by `target_column`.

    Statistics calculated:
        - Mean
        - Standard Deviation (std)
        - Coefficient of Variation (std / mean)
        - Minimum
        - Maximum
        - Median
        - 25th, 50th, 75th Percentiles
        - Interquartile Range (IQR = 75th - 25th)

    Error Handling:
        - If a file cannot be read or the target column does not exist, it will be skipped.
        - Any corrupt or empty file will be skipped.

    Returns:
        None
    """
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Pattern to match all CSV files in the input folder
    csv_pattern = os.path.join(input_folder, "*.csv")
    
    results = []
    
    for csv_file in glob.glob(csv_pattern):
        filename = os.path.basename(csv_file)
        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Check if target_column exists
            if target_column not in df.columns:
                print(f"Skipping file '{filename}' - Column '{target_column}' not found.")
                continue
            
            # Drop NaN values in target_column to avoid errors in calculations
            data = df[target_column].dropna()
            
            # If no valid data points remain, skip
            if data.empty:
                print(f"Skipping file '{filename}' - No valid data in column '{target_column}'.")
                continue
            
            # Compute statistics
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            median_val = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            cv_val = std_val / mean_val if mean_val != 0 else np.nan

            # Almacena los resultados en una lista de dicts para convertirlo luego en DataFrame
            results.append({
                "filename": filename,
                "mean": mean_val,
                "std": std_val,
                "cv": cv_val,  # Coefficient of Variation
                "min": min_val,
                "max": max_val,
                "median": median_val,
                "25%": q1,
                "50%": median_val,  # Igual a la mediana
                "75%": q3,
                "iqr": iqr
            })
        
        except pd.errors.EmptyDataError:
            print(f"Skipping file '{filename}' - Empty or corrupt file.")
        except Exception as e:
            print(f"Skipping file '{filename}' - Error reading file: {str(e)}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid CSV files were processed. No output file will be generated.")
        return
    
    ordered_columns = [
        "filename", "mean", "std", "cv", "min", "max", 
        "median", "25%", "50%", "75%", "iqr"
    ]
    results_df = results_df[ordered_columns]

    output_path = os.path.join(output_folder, f"statistics_profiles_{name_folder}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Statistics have been successfully saved to '{output_path}'.")

    return results_df
