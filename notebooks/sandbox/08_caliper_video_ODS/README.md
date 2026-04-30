# Caliper + Video-log + Ardaman panel builder

Genera un panel comparativo por pozo prioritario que muestra:

- **Panel izquierdo**: senal cruda del caliper con bandas de severidad de
  breakouts coloreadas por muestra (mild / moderate / severe).
- **Panel derecho**: notas del video log y, cuando esta disponible, las
  descripciones litologicas y mediciones de conductividad in-situ del
  reporte Ardaman & Associates de 2009.

Pozos: LRS70D, AW5D, AW6D, BW3D, LRS69D.

## Archivos de entrada


| Archivo | Origen |
|---|---|
| `priority_wells_cumulative_min_v2_perpoint.csv` | pipeline de breakouts |
| `Priority_Ewan_video_logs_v2.xlsx` | trabajo de campo |
| `ardaman_lithology.csv` | transcripcion manual del reporte 2009 |
| `concatenate_caliper_all.csv` | archivo master del caliper (todas los .LAS concatenados) |
| `caliper_videolog_panel.py` | el script |

## Uso

Render de los 5 pozos:

    uv run python .\notebooks\sandbox\08_caliper_video_ODS\caliper_videolog_panel.py

Especificar directorios:

    python caliper_videolog_panel.py --data-dir ./data --out-dir ./figures

Solo algunos pozos:

    python caliper_videolog_panel.py --wells LRS70D AW6D

Las figuras se escriben como `{POZO}_caliper_videolog_panel.png` en el
directorio de salida.

## Notas cientificas importantes

- **El pozo del video y el pozo del caliper no son siempre el mismo**.
  El sufijo del caliper es siempre `D` (deep, perforados en 2021).
  Los videos vienen de:
  - LRS70D -> del mismo pozo D (correlacion directa)
  - AW5D ↔ video AW5O (pozo viejo, perforado 2009)
  - AW6D ↔ video AW6O (idem)
  - BW3D ↔ video BW3S (shallow , perforado 2021)
  - LRS69D ↔ video LRS69S (idem)
  

- **El reporte Ardaman 2009** (`ardaman_lithology.csv`) solo aplica a
  AW5O y AW6O. En el reporte original se llaman B-5 y B-6
  respectivamente. Las mediciones de conductividad in-situ se tomaron
  con un Horiba U-22 durante el coring en marzo de 2009.

- **Las severidades de breakout** se computan en el pipeline upstream
  (`priority_wells_cumulative_min_v2.py`); este script solo consume el
  CSV per-sample. Regla:
  `excess = caliper - (baseline + 1.6 cm + 1*sigma_inst)`,
  con sigma_inst = 0.0881 cm. Cortes:
  mild < 3.2 cm, moderate < 9.6 cm, severe >= 9.6 cm o saturado a
  32.5 cm.

## Estructura del CSV Ardaman

    well, depth_ft, depth_m, kind, text

Donde kind in {lithology, conductivity_in_situ}. Cada linea litologica
representa el tope de una unidad; el script extiende el intervalo hasta
el tope de la siguiente unidad litologica. Las conductividades se
dibujan como punto unico. Lineas con `#` son comentarios.
