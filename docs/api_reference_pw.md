# API Reference: Piecewise Regression Library

## Módulos

### 1. `main`

Contiene las clases principales para ajuste de modelos de regresión segmentada.

#### Clases:

- ``

  - **Descripción:** Permite ajustar un modelo de regresión segmentada mediante el método iterativo de Muggeo con reinicio bootstrap.
  - **Métodos:**
    - `__init__(xx, yy, n_breakpoints=None, start_values=None, n_boot=100, verbose=False, max_iterations=30, tolerance=10**-5, min_distance_between_breakpoints=0.01, min_distance_to_edge=0.02)`
    - `get_results()`: Devuelve los resultados clave del ajuste.
    - `get_params()`: Retorna los parámetros estimados del modelo.
    - `plot_data(**kwargs)`: Gráfica los datos originales.
    - `predict(xx_predict)`: Predice valores de y para una serie de x proporcionada.
    - `plot_fit(**kwargs)`: Gráfica el modelo ajustado.
    - `plot_breakpoints(**kwargs)`: Gráfica las posiciones de los puntos de quiebre.
    - `plot_breakpoint_confidence_intervals(**kwargs)`: Gráfica los intervalos de confianza de los puntos de quiebre.
    - `summary()`: Imprime un resumen del ajuste.

- ``

  - **Descripción:** Implementa el método iterativo de Muggeo para identificar puntos de quiebre.
  - **Métodos:**
    - `__init__(xx, yy, n_breakpoints, start_values=None, verbose=False, max_iterations=30, tolerance=10**-5, min_distance_between_breakpoints=0.01, min_distance_to_edge=0.02)`
    - `fit()`: Ejecuta el procedimiento iterativo.
    - `stop_or_not()`: Determina si el procedimiento debe detenerse.

- ``

  - **Descripción:** Calcula los próximos puntos de quiebre y parámetros intermedios en cada iteración de Muggeo.

### 2. `model_selection`

Contiene herramientas para comparar modelos segmentados y seleccionar el mejor basado en criterios como BIC.

#### Clases:

- ``
  - **Descripción:** Evalúa modelos con diferentes números de puntos de quiebre usando BIC.
  - **Métodos:**
    - `__init__(xx, yy, max_breakpoints=10, n_boot=100, max_iterations=30, tolerance=10**-5, min_distance_between_breakpoints=0.01, min_distance_to_edge=0.02, verbose=True)`
    - `summary()`: Imprime un resumen comparativo de los modelos ajustados.

### 3. `davies`

Implementa la prueba de significancia de Davies para evaluar la existencia de puntos de quiebre.

#### Funciones:

- ``
  - **Descripción:** Evalúa la existencia de puntos de quiebre basado en el cambio de pendiente.
  - **Parámetros:**
    - `xx`: Serie de datos en el eje x.
    - `yy`: Serie de datos en el eje y.
    - `k`: Número de puntos a evaluar.
    - `alternative`: Tipo de prueba ("two\_sided", "less", "greater").

### 4. `r_squared_calc`

Cálculo del coeficiente de determinación R² y R² ajustado.

#### Funciones:

- ``
  - **Descripción:** Calcula los valores de R², R² ajustado, y las sumas de cuadrados residuales y totales.
  - **Parámetros:**
    - `yy`: Valores reales de y.
    - `ff`: Valores ajustados de y.
    - `n_params`: Número de parámetros del modelo.

### 5. `data_validation`

Validaciones de entrada para los parámetros utilizados en los modelos.

#### Funciones:

- ``: Verifica si una variable es booleana.
- ``: Valida si una variable es un entero positivo.
- ``: Verifica si es un entero no negativo.
- ``: Valida si es un número positivo.
- ``: Comprueba si es una lista de números con una longitud mínima especificada.

## Inicialización del Paquete

En el archivo `__init__.py` se definen las exportaciones principales:

```python
from .main import Fit, Muggeo
from .model_selection import ModelSelection
from .davies import davies_test

__all__ = ['Fit', 'Muggeo', 'ModelSelection', 'davies_test']
```

Este paquete está diseñado para facilitar el ajuste y análisis de modelos de regresión segmentada, con capacidades de evaluación de modelos y pruebas de significancia.

