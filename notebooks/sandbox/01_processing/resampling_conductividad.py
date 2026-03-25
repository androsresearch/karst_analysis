import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import glob
from typing import Dict, Any, Optional
import plotly.graph_objects as go

# =========================
# CONFIGURACIÓN
# =========================
csv_path = r"C:\Users\Mariana\Documents\freshwater_lens\data\rawdy\rawdy_sat51w2p_lrs70\LRS70_D_YSI_R_20250226_processed.csv"

depth_col = "Vertical Position m"
cond_col = "Corrected sp Cond [µS/cm]"

depth_bin_size = 1.0        # metros
cond_bin_size = 200.0       # µS/cm

# =========================
# LEER ARCHIVO
# =========================
df = pd.read_csv(csv_path)

# Verificar columnas
required_cols = [depth_col, cond_col]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"No se encontró la columna requerida: {col}")

# Limpiar datos
df = df[[depth_col, cond_col]].dropna().copy()

# =========================
# FUNCIÓN PARA CONTAR DENSIDAD POR BINS
# =========================
def make_bin_counts(series, bin_size):
    """
    Calcula conteos por intervalos (bins) comenzando desde el valor mínimo
    y avanzando en incrementos de bin_size.
    """
    min_val = series.min()
    max_val = series.max()

    # Asegura cubrir completamente el máximo
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Si por alguna razón solo hubiera un bin, agregar otro
    if len(bins) < 2:
        bins = np.array([min_val, min_val + bin_size])

    counts, edges = np.histogram(series, bins=bins)

    bin_table = pd.DataFrame({
        "bin_start": edges[:-1],
        "bin_end": edges[1:],
        "count": counts
    })

    bin_table["bin_label"] = [
        f"[{start:.3f}, {end:.3f})" for start, end in zip(bin_table["bin_start"], bin_table["bin_end"])
    ]

    return bin_table, edges

# =========================
# RESAMPLEAR POR CONDUCTIVIDAD
# =========================

def resample_uniform_conductivity(
    df,
    cond_col="Corrected sp Cond [µS/cm]",
    depth_col="Vertical Position m",
    bin_size=200,
    n_per_bin=50,
    method="median",  # "random" o "mean"
    random_state=42
):
    """
    Rebalancea los datos para tener densidad uniforme en conductividad.
    """

    df = df[[depth_col, cond_col]].dropna().copy()

    # Crear bins en conductividad
    min_c = df[cond_col].min()
    max_c = df[cond_col].max()

    bins = np.arange(min_c, max_c + bin_size, bin_size)
    df["cond_bin"] = pd.cut(df[cond_col], bins=bins, include_lowest=True)

    resampled = []

    for _, group in df.groupby("cond_bin"):
        if len(group) == 0:
            continue

        if method == "random":
            # Downsample o upsample
            if len(group) >= n_per_bin:
                sampled = group.sample(n=n_per_bin, random_state=random_state)
            else:
                sampled = group.sample(n=n_per_bin, replace=True, random_state=random_state)

        elif method == "median":
            # Representar el bin con un punto promedio
            sampled = pd.DataFrame({
                depth_col: [group[depth_col].median()],
                cond_col: [group[cond_col].median()]
            })

        resampled.append(sampled)

    df_resampled = pd.concat(resampled, ignore_index=True)

    return 

# =========================
# EJECUTAR
# =========================


df_resampled = resample_uniform_conductivity(df)
depth_counts, depth_edges = make_bin_counts(df_resampled[depth_col], depth_bin_size) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cond_counts, cond_edges = make_bin_counts(df_resampled[cond_col], cond_bin_size)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df_resampled.to_csv(r'C:\Users\Mariana\Documents\freshwater_lens\data\rawdy\rawdy_satd_secuniform_lrs70\LRS70_D_YSI_R_20250226_secuniform.csv')

# =========================
# FIGURA 1.DENSIDAD POR PROFUNDIDAD
# =========================


fig_depth = go.Figure(
    data=[
        go.Bar(
            x=depth_counts["bin_start"],
            y=depth_counts["count"],
            customdata=np.stack(
                [depth_counts["bin_start"], depth_counts["bin_end"]], axis=-1
            ),
            hovertemplate=(
                "Intervalo de profundidad: [%{customdata[0]:.3f}, %{customdata[1]:.3f}) m<br>"
                "Número de puntos: %{y}<extra></extra>"
            )
        )
    ]
)

fig_depth.update_layout(
    title="Densidad de puntos en función de la profundidad",
    xaxis_title="Profundidad (m)",
    yaxis_title="Número de puntos",
    bargap=0.05,
    template="plotly_white"
)

# =========================
# FIGURA 2. DENSIDAD POR CONDUCTIVIDAD
# =========================

fig_cond = go.Figure(
    data=[
        go.Bar(
            x=cond_counts["bin_start"],
            y=cond_counts["count"],
            customdata=np.stack(
                [cond_counts["bin_start"], cond_counts["bin_end"]], axis=-1
            ),
            hovertemplate=(
                "Intervalo de conductividad: [%{customdata[0]:.2f}, %{customdata[1]:.2f}) µS/cm<br>"
                "Número de puntos: %{y}<extra></extra>"
            )
        )
    ]
)

fig_cond.update_layout(
    title="Densidad de puntos en función de la conductividad",
    xaxis_title="Conductividad (µS/cm)",
    yaxis_title="Número de puntos",
    bargap=0.05,
    template="plotly_white"
)

# =========================
# MOSTRAR RESULTADOS
# =========================
print("Resumen profundidad:")
print(depth_counts)

print("\nResumen conductividad:")
print(cond_counts)

fig_depth.show()
fig_cond.show()