
> [!NOTE] 
**Nota para Mar**: Los flujos de datos no soy muy elegantes, pero funcionan jsjs :D


## Instalación

1. Clona el repositorio
    ```sh
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2. Crea un entorno virtual

    ```sh
    python -m venv venv
    ```

4. Activa el entorno virtual
    - En Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - En macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

5. Instala las dependencias del 
    ```sh
    pip install -r requirements.txt
    ```

6. LLena la carpeta `data/row` con todos los archivos `.csv` a procesar

7. ¡Haz que la mágia suceda!, ejecuta en la terminal (desde el root): 

```sh
python main.py
```

> [!NOTE]
**Información:** Puedes modificar los parámetros desde `main.py` del filtro Savitzky-Golay y los parámetro para el cálculo de ajuste segmentado y BIC.

> [!WARNING]
**Atención:** Debido al nivel de calculos realizados, procesar todos los datos demorará bastante tiempo. Asegúrate de estar preparado para ejecutar este comando. 

## Métricas y gráficas

> Para consultar las métricas de evaluación ($R^2$, $R^2$ ajustada, RMS, localización de breakpoints) y visualuzar los resultados, debes ejecutar el notebook `notebooks/evaluation.ipynb`. El único `input` es el nombre (sin extensión) del archivo/perfil que quieras evaluar y visualizar :). 

> Se añade en la data un perfil de ejemplo (`BW5D_YSI_20230822`) para que puedas probar dicho notebook :) 


## Boxplots 

1. Cargar todos los perfiles `data/rawdy` y CSV `fwl_2024_TW.csv` en `data/fwl_map`.
2. Ejecutar `notebooks/filter_statistics.ipynb`