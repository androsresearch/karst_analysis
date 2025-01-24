import pandas as pd

def load_data(filepath:str, 
            x_col:str='Vertical Position m', y_col:str='Corrected sp Cond [µS/cm]',
            json:bool=False
            ):
    """
    Carga datos desde un archivo CSV y retorna dos arreglos: x y y.

    Parámetros:
    -----------
    filepath : str
        Ruta al archivo CSV.
    x_col : str, opcional
        Nombre de la columna que se usará como eje x.
    y_col : str, opcional
        Nombre de la columna que se usará como eje y.
    json : bool, opcional
        Indica si el archivo es un JSON.

    Retorna:
    --------
    x : np.ndarray
        Array con los valores de la columna x_col.
    y : np.ndarray
        Array con los valores de la columna y_col.
    """
    if json:
        df = pd.read_json(filepath)
        return df
    else:
        df = pd.read_csv(filepath)

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Las columnas '{x_col}' o '{y_col}' no existen en el archivo CSV.")

        x = df[x_col].values
        y = df[y_col].values
    
        return x, y