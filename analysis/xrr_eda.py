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

"""Reusable helpers for XRR exploratory data analysis notebooks."""
from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


DEFAULT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Helvetica", size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=80, r=35, t=60, b=65),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#cccccc",
        borderwidth=0.6,
    ),
)


def _merge_layout(layout: dict | None) -> dict:
    merged = DEFAULT_LAYOUT.copy()
    if layout:
        merged.update(layout)
    return merged


def _decode_value(value):
    """Gracefully convert HDF5 scalars/arrays to native Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "dtype") and value.dtype.kind in {"S", "O"}:
        flat = value.ravel().tolist()
        return [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v for v in flat]
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.item()
    except Exception:
        return value


def load_xrr_hdf5(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load reflectivity curves, fit parameters, and metadata into tidy DataFrames."""
    intensity_records: list[dict] = []
    fit_records: list[dict] = []
    metadata_records: list[dict] = []

    with h5py.File(path, "r") as h5:
        for sample in h5:
            grp = h5[sample]
            exp = grp["experiment"]
            data = np.atleast_2d(np.array(exp["data"]))
            q = np.array(exp["q"])
            scan_values = exp["scan"][()]
            scan_array = np.atleast_1d(scan_values)
            scan_array = np.array([_decode_value(v) for v in np.atleast_1d(scan_array)])
            scan_array = (
                np.array([int(v) for v in scan_array], dtype=int)
                if scan_array.size
                else np.arange(data.shape[0])
            )
            if scan_array.shape[0] != data.shape[0]:
                scan_array = np.arange(data.shape[0])

            for idx, scan_id in enumerate(scan_array):
                scan_intensity = data[idx]
                for q_val, refl in zip(q, scan_intensity):
                    intensity_records.append(
                        {
                            "sample": sample,
                            "scan_id": int(scan_id),
                            "q": float(q_val),
                            "reflectivity": float(refl),
                        }
                    )

            if "fit" in grp:
                fit_grp = grp["fit"]
                film_thickness = (
                    np.atleast_1d(np.array(fit_grp.get("Film_thickness")))
                    if "Film_thickness" in fit_grp
                    else None
                )
                film_roughness = (
                    np.atleast_1d(np.array(fit_grp.get("Film_roughness")))
                    if "Film_roughness" in fit_grp
                    else None
                )
                num_scans = data.shape[0]
                for i in range(num_scans):
                    thickness_val = (
                        float(film_thickness[min(i, film_thickness.size - 1)])
                        if film_thickness is not None and film_thickness.size
                        else np.nan
                    )
                    roughness_val = (
                        float(film_roughness[min(i, film_roughness.size - 1)])
                        if film_roughness is not None and film_roughness.size
                        else np.nan
                    )
                    fit_records.append(
                        {
                            "sample": sample,
                            "scan_id": int(scan_array[i]) if scan_array.size else i,
                            "film_thickness_nm": thickness_val,
                            "film_roughness_nm": roughness_val,
                        }
                    )

            if "metadata" in grp:
                meta = {key: _decode_value(ds[()]) for key, ds in grp["metadata"].items()}
                meta["sample"] = sample
                metadata_records.append(meta)

    intensity_df = pd.DataFrame(intensity_records)
    fit_df = (
        pd.DataFrame(fit_records)
        if fit_records
        else pd.DataFrame(columns=["sample", "scan_id", "film_thickness_nm", "film_roughness_nm"])
    )
    metadata_df = (
        pd.DataFrame(metadata_records).set_index("sample")
        if metadata_records
        else pd.DataFrame()
    )
    return intensity_df, fit_df, metadata_df


def summarise_samples(
    intensity_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate scan statistics and merge optional fit/metadata tables."""
    intensity_df = intensity_df.copy()
    intensity_df.sort_values(["sample", "scan_id", "q"], inplace=True)

    thickness_stats = (
        fit_df.groupby("sample")["film_thickness_nm"].agg(["mean", "std"])
        if not fit_df.empty
        else pd.DataFrame(columns=["mean", "std"])
    )
    thickness_stats.rename(
        columns={"mean": "thickness_mean_nm", "std": "thickness_std_nm"}, inplace=True
    )

    metadata_view = metadata_df.copy()
    if not metadata_view.empty and metadata_view.index.name != "sample":
        metadata_view.index.name = "sample"

    sample_summary = (
        intensity_df.groupby("sample")
        .agg(
            scans=("scan_id", "nunique"),
            q_min=("q", "min"),
            q_max=("q", "max"),
            median_reflectivity=("reflectivity", "median"),
        )
        .join(thickness_stats)
        .join(metadata_view, how="left")
        .reset_index()
    )
    return sample_summary, metadata_view


def style_sample_summary(sample_summary: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply publication-style formatting to the per-sample summary table."""
    def _format_year(value):
        return "—" if pd.isna(value) else f"{int(value):d}"

    formatters: dict[str, callable] = {
        "q_min": "{:.3f}".format,
        "q_max": "{:.3f}".format,
        "median_reflectivity": "{:.3e}".format,
    }
    if "thickness_mean_nm" in sample_summary.columns:
        formatters["thickness_mean_nm"] = "{:.1f}".format
    if "thickness_std_nm" in sample_summary.columns:
        formatters["thickness_std_nm"] = "{:.1f}".format
    if "year_experiment" in sample_summary.columns:
        sample_summary = sample_summary.copy()
        sample_summary["year_experiment"] = pd.to_numeric(
            sample_summary["year_experiment"], errors="coerce"
        )
        formatters["year_experiment"] = _format_year

    return sample_summary.style.format(formatters, na_rep="—").set_caption(
        "Per-sample reflectivity coverage summarised from experimental scans."
    )


def build_coverage_figure(sample_summary: pd.DataFrame, layout: dict | None = None) -> go.Figure:
    """Create the dual-axis figure summarising scan counts and q-span."""
    merged_layout = _merge_layout(layout)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=sample_summary["sample"],
            y=sample_summary["scans"],
            name="Number of scans",
            marker_color="#264653",
        ),
        secondary_y=False,
    )

    valid_mask = sample_summary[["q_min", "q_max"]].notna().all(axis=1)
    if valid_mask.any():
        mid_q = (sample_summary["q_min"] + sample_summary["q_max"]) / 2
        q_upper = sample_summary["q_max"] - mid_q
        q_lower = mid_q - sample_summary["q_min"]
        fig.add_trace(
            go.Scatter(
                x=sample_summary.loc[valid_mask, "sample"],
                y=mid_q.loc[valid_mask],
                mode="markers",
                name="q-range midpoint",
                marker=dict(color="#e76f51", size=10, symbol="circle"),
                error_y=dict(
                    type="data",
                    array=q_upper.loc[valid_mask],
                    arrayminus=q_lower.loc[valid_mask],
                    thickness=1.4,
                ),
            ),
            secondary_y=True,
        )

    layout_kwargs = {k: v for k, v in merged_layout.items() if k != "legend"}
    fig.update_layout(
        **layout_kwargs,
        title="Experimental Coverage by Sample",
        xaxis=dict(
            title="Sample",
            tickangle=-35,
            ticks="outside",
            ticklen=6,
            tickwidth=1.2,
            linecolor="#000",
            mirror=False,
        ),
        yaxis=dict(
            title="Scan count",
            ticks="outside",
            ticklen=6,
            tickwidth=1.2,
            linecolor="#000",
            mirror=True,
        ),
        yaxis2=dict(title="Momentum transfer q midpoint (Å⁻¹)"),
    )
    legend_layout = merged_layout.get("legend", {}).copy()
    legend_layout.update(y=1.15)
    fig.update_layout(legend=legend_layout)
    return fig


def build_grouped_profiles_widget(
    intensity_df: pd.DataFrame,
    metadata_view: pd.DataFrame,
    layout: dict | None = None,
    palette: Iterable[str] | None = None,
) -> widgets.VBox:
    """Return an interactive widget for grouping/filtering median reflectivity curves."""
    merged_layout = _merge_layout(layout)
    palette_colors = list(palette) if palette is not None else list(qualitative.D3 + qualitative.Safe)

    avg_reflectivity = (
        intensity_df.groupby(["sample", "q"])["reflectivity"].median().reset_index()
    )
    if not metadata_view.empty:
        avg_reflectivity = avg_reflectivity.merge(
            metadata_view.reset_index(),
            on="sample",
            how="left",
        )

    group_fields = [("Sample", "sample")]
    for col in metadata_view.columns:
        group_fields.append((col.replace("_", " ").title(), col))
    available_fields = [(label, field) for label, field in group_fields if field in avg_reflectivity.columns]
    label_lookup = {field: label for label, field in available_fields}

    group_dropdown = widgets.Dropdown(
        options=available_fields,
        value=available_fields[0][1],
        description="Group by",
        layout=widgets.Layout(width="260px"),
    )
    selection_widget = widgets.SelectMultiple(
        options=(),
        value=(),
        description="Filter",
        layout=widgets.Layout(width="260px", height="210px"),
    )

    fig_widget = go.FigureWidget(
        layout=go.Layout(
            **merged_layout,
            title="Median Reflectivity Profiles",
            xaxis=dict(
                title="Momentum transfer q (Å⁻¹)",
                ticks="outside",
                ticklen=6,
                tickwidth=1.2,
                linecolor="#000",
                mirror=True,
            ),
            yaxis=dict(
                title="Reflectivity (arb. units)",
                type="log",
                ticks="outside",
                ticklen=6,
                tickwidth=1.2,
                linecolor="#000",
                mirror=True,
            ),
        )
    )

    def _refresh_selection_options(group_field: str):
        labels = avg_reflectivity[group_field].dropna().astype(str).unique().tolist()
        labels.sort()
        selection_widget.options = labels
        selection_widget.value = ()

    def _update_profiles(*_):
        group_field = group_dropdown.value
        if group_field not in avg_reflectivity.columns:
            return
        grouped = (
            avg_reflectivity.groupby([group_field, "q"], dropna=True)["reflectivity"].median().reset_index()
        )
        selected = list(selection_widget.value)
        if selected:
            grouped = grouped[grouped[group_field].astype(str).isin(selected)]

        color_cycle = cycle(palette_colors)
        traces = []
        for label, subset in grouped.groupby(group_field):
            subset = subset.sort_values("q")
            traces.append(
                go.Scatter(
                    x=subset["q"],
                    y=subset["reflectivity"],
                    mode="lines",
                    name=str(label),
                    line=dict(width=2.2, color=next(color_cycle)),
                )
            )
        with fig_widget.batch_update():
            fig_widget.data = tuple()
            for trace in traces:
                fig_widget.add_trace(trace)
            legend_title = label_lookup.get(group_field, group_field.title())
            fig_widget.layout.title = f"Median Reflectivity Profiles by {legend_title}"
            fig_widget.layout.legend.title = legend_title

    def _on_group_change(change):
        if change.get("name") != "value":
            return
        _refresh_selection_options(change["new"])
        _update_profiles()

    def _on_filter_change(change):
        if change.get("name") != "value":
            return
        _update_profiles()

    group_dropdown.observe(_on_group_change, names="value")
    selection_widget.observe(_on_filter_change, names="value")

    _refresh_selection_options(group_dropdown.value)
    _update_profiles()

    controls = widgets.HBox([group_dropdown, selection_widget])
    return widgets.VBox([controls, fig_widget])


def build_sample_surface_widget(
    intensity_df: pd.DataFrame,
    sample_summary: pd.DataFrame,
    layout: dict | None = None,
) -> widgets.VBox:
    """Return a dropdown + 3D surface widget for scan-to-scan variability."""
    merged_layout = _merge_layout(layout)
    sample_options = sample_summary["sample"].tolist()
    default_sample = sample_options[0] if sample_options else None

    def _surface_arrays_for_sample(sample_label: str):
        if sample_label is None:
            return np.array([[0.0]]), np.array([[0.0]]), np.array([[np.nan]])
        sample_data = intensity_df[intensity_df["sample"] == sample_label]
        if sample_data.empty:
            return np.array([[0.0]]), np.array([[0.0]]), np.array([[np.nan]])
        heatmap_data = sample_data.pivot_table(
            index="scan_id",
            columns="q",
            values="reflectivity",
            aggfunc="mean",
        )
        heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
        if heatmap_data.empty:
            return np.array([[0.0]]), np.array([[0.0]]), np.array([[np.nan]])
        Z = np.log10(heatmap_data.values + 1e-12)
        X, Y = np.meshgrid(heatmap_data.columns.to_numpy(), heatmap_data.index.to_numpy())
        return X, Y, Z

    selector = widgets.Dropdown(
        options=sample_options,
        value=default_sample,
        description="Sample",
        layout=widgets.Layout(width="280px"),
    )

    X0, Y0, Z0 = _surface_arrays_for_sample(selector.value)
    layout_kwargs = {k: v for k, v in merged_layout.items() if k not in {"legend", "margin"}}
    fig_widget = go.FigureWidget(
        data=[
            go.Surface(
                x=X0,
                y=Y0,
                z=Z0,
                colorscale="Viridis",
                colorbar=dict(title="log10 R"),
                showscale=True,
            )
        ],
        layout=go.Layout(
            **layout_kwargs,
            title=(
                f"Scan-to-scan Variability Surface for {selector.value}"
                if selector.value
                else "Scan-to-scan Variability Surface"
            ),
            scene=dict(
                xaxis_title="Momentum transfer q (Å⁻¹)",
                yaxis_title="Scan index",
                zaxis_title="log10 Reflectivity",
                xaxis=dict(backgroundcolor="white", gridcolor="#cccccc", showspikes=False),
                yaxis=dict(backgroundcolor="white", gridcolor="#cccccc", showspikes=False),
                zaxis=dict(backgroundcolor="white", gridcolor="#cccccc"),
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        ),
    )

    def _on_sample_change(change):
        if change.get("name") != "value" or change.get("new") is None:
            return
        X, Y, Z = _surface_arrays_for_sample(change["new"])
        with fig_widget.batch_update():
            fig_widget.data[0].x = X
            fig_widget.data[0].y = Y
            fig_widget.data[0].z = Z
            fig_widget.layout.title = f"Scan-to-scan Variability Surface for {change['new']}"

    selector.observe(_on_sample_change, names="value")
    return widgets.VBox([selector, fig_widget])
