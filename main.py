"""
Script principal para:
1. Cargar y procesar automáticamente todos los archivos .csv en la carpeta ./data/row.
2. Aplicar las funciones de procesamiento y análisis en el orden especificado.
3. Manejar errores y generar un informe al finalizar.
"""

import os
import glob
import json
import traceback
import pandas as pd
import time

from modules import load
from modules import processing
from modules import analysis

inicio = time.time()

# Parámetros para el filtro Savitzky-Golay
WINDOW_LENGTH = 25
POLY_ORDER = 2

# Parámetros para calcular puntos de ruptura
MAX_BREAKPOINTS = 10
N_TRIALS = 5


def main():
    """
    Función principal que recorre los archivos .csv en './data/raw' y aplica 
    las distintas funciones de procesamiento. Finalmente, guarda los resultados 
    y genera un informe resumen.
    """
    
    # Directorios de entrada y salida
    input_dir = './data/raw'
    rowdy_dir = './data/rawdy'
    processed_dir = './data/processed'
    results_dir = './data/results'
    
    # Crear los directorios de salida si no existen
    os.makedirs(rowdy_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Listas para llevar el registro de resultados
    processed_files = []
    error_files = []

    # Recorremos todos los archivos .csv dentro de input_dir
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    for csv_file in csv_files:
        # Extraer el nombre base del archivo (sin extensión)
        # Ejemplo: si csv_file = './data/row/ejemplo.csv', name = 'ejemplo'
        name = os.path.splitext(os.path.basename(csv_file))[0]

        try:
            print(f"\nProcesando archivo: {csv_file}")

            # 1. Cargar Datos
            df = load.load_data(csv_file)
            x_row = df[0]
            y_row = df[1]

            # 2. Filtrar Valores Negativos
            x_positive, y_positive = processing.filter_non_negative_values(x_row, y_row)

            # 3. Eliminar Outliers (IQR)
            #x_out, y_out = processing.remove_outliers_iqr(x_positive, y_positive)

            # 4. Promediar Grupos por X
            x_ave, y_ave, duplicates = processing.average_grouped_by_x(x_positive, y_positive)

            # Guardar resultado intermedio (rowdy)
            rowdy_path = os.path.join(rowdy_dir, f"{name}_rowdy.csv")
            _save_to_csv(x_ave, y_ave, rowdy_path)

            # 5. Aplicar Filtro Savitzky-Golay
            y_smoothed = processing.apply_savgol_filter(
                y_ave, 
                window_length=WINDOW_LENGTH, 
                poly_order=POLY_ORDER
            )

            # Guardar resultado intermedio (processed)
            processed_path = os.path.join(processed_dir, f"{name}_processed.csv")
            _save_to_csv(x_ave, y_smoothed, processed_path)

            # 6. Calcular Mejores Puntos de Ruptura
            results = analysis.best_n_breakpoints(
                x_ave, 
                y_smoothed, 
                max_breakpoints=MAX_BREAKPOINTS, 
                n_trials=N_TRIALS
            )

            results = pd.DataFrame(results)

            # Guardar resultados en JSON
            results_path = os.path.join(results_dir, f"{name}_results.json")
            _save_to_json(results, results_path)
            
            # Si todo salió bien, lo agregamos a la lista de procesados
            processed_files.append(name)

        except Exception as e:
            # Manejo de errores: guardar la traza y continuar con el siguiente archivo
            error_files.append({
                'file': name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            print(f"ERROR procesando {csv_file}: {e}")
            print("Saltando al siguiente archivo...\n")
            continue
    
    # Generar informe resumen
    _print_summary(processed_files, error_files)
    print("El timepo de ejecucuón fue de: ", time.time()-inicio, "segundos")


def _save_to_csv(x_values, y_values, file_path, x_label="Vertical Position [m]", y_label="Corrected sp Cond [uS/cm]"):
    """
    Guarda dos listas (x_values, y_values) en un archivo CSV con las etiquetas
    de columna especificadas.
    
    :param x_values: Lista de valores de la columna X.
    :param y_values: Lista de valores de la columna Y.
    :param file_path: Ruta donde guardar el archivo CSV.
    :param x_label: Etiqueta para la columna X.
    :param y_label: Etiqueta para la columna Y.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Escribir la cabecera con los nuevos nombres de columnas
            f.write(f"{x_label},{y_label}\n")
            # Escribir los datos
            for x, y in zip(x_values, y_values):
                f.write(f"{x},{y}\n")
    except Exception as e:
        raise IOError(f"No se pudo guardar el CSV en {file_path}: {e}")


def _save_to_json(data, file_path):
    """
    Guarda el objeto `data` en un archivo JSON utilizando la función nativa de pandas.
    """
    try:
        # Si el objeto es un DataFrame, usa su método nativo `to_json`
        if isinstance(data, pd.DataFrame):
            data.to_json(file_path)
        else:
            # Para cualquier otro tipo de dato, usar json.dump como antes
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"No se pudo guardar el JSON en {file_path}: {e}")


def _print_summary(processed_files, error_files):
    """
    Imprime un resumen final con los archivos procesados exitosamente y 
    los que causaron errores.
    """
    print("\n===== RESUMEN DEL PROCESO =====")
    print(f"Total de archivos procesados: {len(processed_files) + len(error_files)}\n")

    if processed_files:
        print("Archivos procesados exitosamente:")
        for pf in processed_files:
            print(f"  - {pf}")
    else:
        print("No se procesó ningún archivo exitosamente.")

    if error_files:
        print("\nArchivos con errores:")
        for ef in error_files:
            print(f"  - {ef['file']}")
            print(f"    Error: {ef['error']}")
    else:
        print("\nNo hubo errores en el procesamiento.")


if __name__ == '__main__':
    main()