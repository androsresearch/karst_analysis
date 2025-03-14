from piecewise_regression import Fit
import numpy as np

def rebuild_model(xx, yy, params_dict, tolerance=1e-5, min_distance=0.01):
    """
    Reconstruye un modelo `Fit` a partir de un diccionario de parámetros.

    :param xx: Lista o array con los datos independientes originales.
    :param yy: Lista o array con los datos dependientes originales.
    :param params_dict: Diccionario de parámetros generado por ModelSelection.
        Debe contener:
            - "n_breakpoints": Número de puntos de quiebre.
            - "estimates": Parámetros estimados (incluyendo breakpoints).

    :return: Objeto `Fit` reconstruido.
    """
    # Validar si el diccionario tiene los parámetros necesarios
    if "n_breakpoints" not in params_dict or "estimates" not in params_dict:
        raise ValueError("El diccionario debe contener 'n_breakpoints' y 'estimates'.")

    # Extraer el número de puntos de quiebre y sus posiciones
    n_breakpoints = params_dict["n_breakpoints"]
    estimates = params_dict["estimates"]

    # Obtener las posiciones de los breakpoints desde los valores de 'estimate'
    breakpoints = [estimates[f"breakpoint{i+1}"]["estimate"] for i in range(n_breakpoints)]

    # Crear el modelo `Fit` utilizando los breakpoints extraídos
    rebuilt_model = Fit(xx, yy, start_values=breakpoints, n_breakpoints=n_breakpoints, n_boot=0, tolerance=tolerance, min_distance_between_breakpoints=min_distance)

    return rebuilt_model

def get_breakpoint_data(data, n_breakpoints):
    """
    Extrae los datos asociados con un número específico de breakpoints.

    :param data: Diccionario con las claves 'bic', 'n_breakpoints', y 'estimates'.
    :param n_breakpoints: Número de breakpoints como entero o string.
    :return: Diccionario con los valores de 'bic', 'n_breakpoints' y 'estimates' para el breakpoint dado.
    """
    key = str(n_breakpoints)  # Convierte el número a string para buscar en el diccionario
    if key in data['n_breakpoints']:
        return {
            'bic': data['bic'][key],
            'n_breakpoints': data['n_breakpoints'][key],
            'estimates': data['estimates'][key]
        }
    else:
        return f"No se encontró información para n_breakpoints = {n_breakpoints}"

def extract_segments(fit_object):
    """
    Extract segment information and fitted model details from a Fit object.

    Parameters:
    - fit_object (Fit): An instance of the `Fit` class from `piecewise_regression`.

    Returns:
    - dict: A dictionary containing segment data and fitted model information.
    
    Example:
    ```python
    segments_info = extract_segments(fit)
    ```
    """
    if not hasattr(fit_object, 'best_muggeo') or not fit_object.best_muggeo:
        raise ValueError("The Fit object did not converge or does not have a valid model.")

    best_fit = fit_object.best_muggeo.best_fit
    segments = []

    # Breakpoints
    breakpoints = best_fit.next_breakpoints
    xx = np.array(fit_object.xx)
    yy = np.array(fit_object.yy)

    # Add edges to breakpoints for segment division
    all_edges = [min(xx)] + list(breakpoints) + [max(xx)]

    for i in range(len(all_edges) - 1):
        segment_start = all_edges[i]
        segment_end = all_edges[i + 1]

        # Filter data within the current segment
        segment_mask = (xx >= segment_start) & (xx <= segment_end)
        segment_x = xx[segment_mask]
        segment_y = yy[segment_mask]

        # Calculate fitted values considering breakpoints and parameters
        intercept = best_fit.raw_params[0]
        alpha = best_fit.raw_params[1]
        beta_hats = best_fit.raw_params[2:2 + len(breakpoints)]

        y_fit = intercept + alpha * segment_x
        for j, bp in enumerate(breakpoints):
            y_fit += beta_hats[j] * np.maximum(0, segment_x - bp)

        segments.append({
            "segment": i + 1,
            "data_x": segment_x,
            "data_y": segment_y,
            "fitted_model": {
                "slope": alpha,
                "intercept": intercept,
                "fitted_y": y_fit
            }
        })

    return {"segments": segments}