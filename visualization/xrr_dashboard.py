# MIT License
#
# Copyright (c) 2024 XRR Demo Code
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Interactive widgets for exploring the XRR forward model."""
from __future__ import annotations

import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets.embed import dependency_state, embed_minimal_html

from core.xrr_forward_model import simulate_reflectivity_curve

__all__ = ["interactive_reflectivity_view", "export_dashboard_html"]

COLOR_CYCLE = ["#264653", "#2a9d8f", "#e9c46a"]
SLIDER_STYLE = {"description_width": "155px"}
SLIDER_LAYOUT = widgets.Layout(width="100%")


def interactive_reflectivity_view():
    """Return an ipywidgets VBox with configuration controls and the interactive plots."""
    layer_controls = []
    for idx in range(3):
        thickness = widgets.FloatSlider(
            value=12.0 if idx == 0 else 6.0,
            min=0.0,
            max=200.0,
            step=0.5,
            description="Thickness (nm)",
            continuous_update=False,
            style=SLIDER_STYLE,
            layout=SLIDER_LAYOUT,
        )
        density = widgets.FloatSlider(
            value=2.2,
            min=1.0,
            max=12.0,
            step=0.05,
            description="Density (g/cm³)",
            continuous_update=False,
            style=SLIDER_STYLE,
            layout=SLIDER_LAYOUT,
        )
        roughness = widgets.FloatSlider(
            value=0.3,
            min=0.0,
            max=3.0,
            step=0.05,
            description="Roughness (nm)",
            continuous_update=False,
            style=SLIDER_STYLE,
            layout=SLIDER_LAYOUT,
        )
        section = widgets.VBox([thickness, density, roughness], layout=widgets.Layout(padding="0 0 10px 0"))
        layer_controls.append((thickness, density, roughness, section))

    substrate_density = widgets.FloatSlider(
        value=2.33,
        min=1.0,
        max=12.0,
        step=0.05,
        description="Substrate density (g/cm³)",
        continuous_update=False,
        style=SLIDER_STYLE,
        layout=SLIDER_LAYOUT,
    )
    substrate_roughness = widgets.FloatSlider(
        value=0.4,
        min=0.0,
        max=3.0,
        step=0.05,
        description="Substrate roughness (nm)",
        continuous_update=False,
        style=SLIDER_STYLE,
        layout=SLIDER_LAYOUT,
    )
    two_theta_max = widgets.FloatSlider(
        value=6.0,
        min=1.0,
        max=10.0,
        step=0.1,
        description="2θ max (°)",
        continuous_update=False,
        style=SLIDER_STYLE,
        layout=SLIDER_LAYOUT,
    )
    info = widgets.HTML("<b>Tip:</b> set thickness to 0 nm to omit a layer.")

    accordion_children = [ctrls[-1] for ctrls in layer_controls]
    layer_accordion = widgets.Accordion(children=accordion_children)
    for idx in range(3):
        layer_accordion.set_title(idx, f"Layer {idx + 1}")

    controls_column = widgets.VBox(
        [info, layer_accordion, substrate_density, substrate_roughness, two_theta_max],
        layout=widgets.Layout(width="360px", padding="0 18px 0 0"),
    )

    stack_fig = go.FigureWidget()
    stack_fig.update_layout(
        template="plotly_white",
        height=320,
        width=640,
        margin=dict(l=80, r=40, t=60, b=60),
        title="Layer Architecture",
        font=dict(family="Helvetica", size=12),
    )
    stack_fig.update_xaxes(visible=False, range=[0, 1], showgrid=False)
    stack_fig.update_yaxes(title_text="Depth (nm)", autorange="reversed")

    reflect_fig = go.FigureWidget()
    reflect_fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=90, r=40, t=70, b=80),
        title=dict(text="Simulated Specular Reflectivity", x=0.02, xanchor="left"),
        font=dict(family="Helvetica", size=13),
        width=1020,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=0.6,
        ),
    )
    reflect_fig.update_xaxes(
        title_text="2θ (degrees)",
        showgrid=True,
        gridcolor="#d9d9d9",
        zeroline=False,
        ticks="outside",
        ticklen=6,
        tickwidth=1.2,
        linecolor="#000000",
        mirror=True,
    )
    reflect_fig.update_yaxes(
        title_text="Reflectivity (arb. units)",
        type="log",
        range=[-9.5, 0],
        showgrid=True,
        gridcolor="#d9d9d9",
        ticks="outside",
        ticklen=6,
        tickwidth=1.2,
        linecolor="#000000",
        mirror=True,
        tickvals=[1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        ticktext=["1e0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"],
    )

    def collect_layers():
        layers = []
        for idx, (thickness_ctrl, density_ctrl, roughness_ctrl, _) in enumerate(layer_controls):
            if thickness_ctrl.value <= 0.0:
                continue
            layers.append(
                {
                    "name": f"Layer {idx + 1}",
                    "thickness_nm": thickness_ctrl.value,
                    "density": density_ctrl.value,
                    "roughness_nm": roughness_ctrl.value,
                }
            )
        return layers

    def update(*_):
        layers = collect_layers()
        params = {
            "layers": layers,
            "substrate_density": substrate_density.value,
            "substrate_roughness_nm": substrate_roughness.value,
            "two_theta_max": two_theta_max.value,
        }
        two_theta, reflectivity = simulate_reflectivity_curve(**params)
        total_thickness = sum(layer["thickness_nm"] for layer in layers)

        with stack_fig.batch_update():
            stack_fig.data = []
            stack_fig.layout.annotations = tuple()
            baseline = max(total_thickness + 15.0, 20.0) if layers else 20.0
            depth_cursor = 0.0
            if not layers:
                stack_fig.add_scatter(
                    x=[0, 1, 1, 0, 0],
                    y=[0, 0, baseline - 5.0, baseline - 5.0, 0],
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="#e0e0e0",
                    showlegend=False,
                    hovertemplate="No overlayers configured<extra></extra>",
                    hoveron="fills",
                )
            for idx, layer in enumerate(layers):
                depth_next = depth_cursor + layer["thickness_nm"]
                color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                stack_fig.add_scatter(
                    x=[0, 1, 1, 0, 0],
                    y=[depth_cursor, depth_cursor, depth_next, depth_next, depth_cursor],
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=color,
                    opacity=0.8,
                    name=layer["name"],
                    hovertemplate=(
                        f"{layer['name']}<br>t = {layer['thickness_nm']:.2f} nm"
                        f"<br>ρ = {layer['density']:.2f} g/cm³"
                        f"<br>σ = {layer['roughness_nm']:.2f} nm"
                        "<extra></extra>"
                    ),
                    hoveron="fills",
                )
                stack_fig.add_annotation(
                    x=0.04,
                    y=(depth_cursor + depth_next) / 2,
                    text=(
                        f"{layer['name']}<br>t = {layer['thickness_nm']:.1f} nm"
                        f"<br>ρ = {layer['density']:.2f} g/cm³"
                        f"<br>σ = {layer['roughness_nm']:.2f} nm"
                    ),
                    showarrow=False,
                    align="left",
                    font=dict(size=11),
                    bgcolor="rgba(255, 255, 255, 0.85)",
                )
                depth_cursor = depth_next
            stack_fig.add_annotation(
                x=0.5,
                y=depth_cursor + 4.0,
                text=(
                    f"Substrate: ρ = {substrate_density.value:.2f} g/cm³, "
                    f"σ = {substrate_roughness.value:.2f} nm"
                ),
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255, 255, 255, 0.85)",
            )
            stack_fig.update_yaxes(range=[baseline, 0])

        with reflect_fig.batch_update():
            reflect_fig.data = []
            reflect_fig.layout.annotations = tuple()
            reflect_fig.add_scatter(
                x=two_theta,
                y=reflectivity,
                mode="lines",
                line=dict(color="#1a1a1a", width=2.1),
                name="Reflectivity",
                hovertemplate="2θ = %{x:.3f}°<br>R = %{y:.3e}<extra></extra>",
            )
            reflect_fig.update_xaxes(range=[two_theta[0], two_theta[-1]])
            caption = (
                f"Optical thickness = {total_thickness:.1f} nm | "
                f"2θ range = {two_theta[0]:.2f}° – {two_theta[-1]:.2f}°"
                if layers
                else f"2θ range = {two_theta[0]:.2f}° – {two_theta[-1]:.2f}°"
            )
            reflect_fig.add_annotation(
                x=0.02,
                y=1.06,
                xref="paper",
                yref="paper",
                text=caption,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255,255,255,0.85)",
            )

    for thickness, density, roughness, _ in layer_controls:
        for control in (thickness, density, roughness):
            control.observe(update, names="value")
    for control in (substrate_density, substrate_roughness, two_theta_max):
        control.observe(update, names="value")

    update()

    top_row = widgets.HBox(
        [controls_column, stack_fig],
        layout=widgets.Layout(align_items="stretch", width="100%"),
    )
    dashboard = widgets.VBox(
        [top_row, reflect_fig],
        layout=widgets.Layout(width="100%", padding="0 0 10px 0"),
    )
    return dashboard


def export_dashboard_html(widget, output_path="XRR_forward_model_dashboard.html", drop_defaults=True):
    """Write the widget layout to a standalone HTML file with bundled state."""
    if widget is None:
        raise ValueError("Provide the widget returned by interactive_reflectivity_view().")
    state = dependency_state(widget, drop_defaults=drop_defaults)
    embed_minimal_html(
        output_path,
        views=[widget],
        state=state,
        requirejs=False,
        title="XRR Forward Model Dashboard",
        indent=2,
    )
