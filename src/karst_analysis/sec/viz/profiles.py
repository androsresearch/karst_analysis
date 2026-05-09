"""Plotly-based profile plot, kept from the legacy notebooks.

Useful for interactive inspection in notebooks. Not used by the batch
pipeline (which writes static matplotlib PNGs).
"""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_profile_plotly(
    x_values: List[float],
    y_values: List[float],
    plot_mode: str = "lines",
    trace_names: Optional[List[str]] = None,
    secondary_x: Optional[List[float]] = None,
    secondary_y: Optional[List[float]] = None,
    title: str = "",
    x_axis_label: str = "Depth [m]",
    y_axis_label: str = "Specific electrical conductivity [µS/cm]",
    y2_axis_label: str = "Secondary Y-Axis",
    use_secondary_axis: bool = False,
    enable_error_x: bool = False,
    enable_error_y: bool = False,
):
    """Interactive plotly figure of one or two traces. Returns the Figure."""
    trace_names = trace_names or ["Primary", "Secondary"]
    error_x = {"type": "constant", "value": 0.001} if enable_error_x else None
    error_y = {"type": "percent", "value": 0.5} if enable_error_y else None

    fig = (
        make_subplots(specs=[[{"secondary_y": True}]])
        if use_secondary_axis else go.Figure()
    )

    fig.add_trace(
        go.Scatter(
            x=x_values, y=y_values, mode=plot_mode,
            name=f"{trace_names[0]} (n = {len(x_values)})",
            error_x=error_x, error_y=error_y,
        ),
        secondary_y=False if use_secondary_axis else None,
    )

    if secondary_x is not None and secondary_y is not None:
        fig.add_trace(
            go.Scatter(
                x=secondary_x, y=secondary_y, mode=plot_mode,
                name=f"{trace_names[1]} (n = {len(secondary_x)})",
                error_x=error_x, error_y=error_y,
            ),
            secondary_y=True if use_secondary_axis else None,
        )

    fig.update_layout(
        title={"text": title, "y": 0.95, "x": 0.5,
               "xanchor": "center", "yanchor": "top"},
        xaxis_title=x_axis_label,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black", borderwidth=1),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="LightGray",
                     zeroline=True, zerolinewidth=1, zerolinecolor="Gray")
    fig.update_yaxes(title_text=y_axis_label, showgrid=True, gridwidth=0.5,
                     gridcolor="LightGray", zeroline=True, zerolinewidth=1,
                     zerolinecolor="Gray",
                     secondary_y=False if use_secondary_axis else None)
    if use_secondary_axis:
        fig.update_yaxes(title_text=y2_axis_label, showgrid=True, gridwidth=0.5,
                         gridcolor="LightGray", zeroline=True, zerolinewidth=1,
                         zerolinecolor="Gray", secondary_y=True)
    return fig
