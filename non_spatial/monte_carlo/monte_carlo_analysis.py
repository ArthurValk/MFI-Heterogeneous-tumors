"""Monte Carlo simulation analysis and visualization notebook.

marimo edit non_spatial/monte_carlo/monte_carlo_analysis.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt

    # Add project root to path
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.monte_carlo.visualization import MCVisualization

    from non_spatial.parametrization import MetricNames

    return MCVisualization, mo, plt, MetricNames


@app.cell
def _(MCVisualization, mo):
    from tests.test_files.test_monte_carlo import TEST_MC_PATH

    parquet_dir = TEST_MC_PATH

    if not parquet_dir.exists():
        loaded_data = None
        status_msg = mo.md("!! Directory not found")
    else:
        try:
            loaded_data = MCVisualization.load_simulation_results(parquet_dir)
            status_msg = mo.md("Data loaded successfully")
        except Exception as e:
            loaded_data = None
            status_msg = mo.md(f"!! Failed to load: {e}")

    return status_msg, loaded_data


@app.cell
def _(MCVisualization, loaded_data, mo):
    metrics_df = None
    available_times = []

    if loaded_data is not None and "metrics" in loaded_data:
        metrics_df = loaded_data["metrics"]
        available_times = MCVisualization.get_available_times(metrics_df)
        _msg_metrics = mo.md(f"""
        ### Data Summary
        - Timepoints available: {len(available_times)}
        - Time range: {available_times[0]} to {available_times[-1]}
        """)
    else:
        _msg_metrics = mo.md("No metrics data available")
    return _msg_metrics, available_times, metrics_df


@app.cell
def _(mo, available_times):
    time_slider = mo.ui.slider(
        steps=available_times,
        label="Select timepoint for empirical distributions:",
        value=available_times[len(available_times) // 2],
    )
    return time_slider


@app.cell
def _(mo, time_slider):
    mo.md("""Select time point for empirical distributions:""")

    mo.hstack([time_slider])
    return


@app.cell
def _(MCVisualization, metrics_df, plt, time_slider):
    if time_slider is None or metrics_df is None:
        pass
    else:
        selected_time = time_slider.value
        MCVisualization.plot_metric_distributions_at_time(
            metrics_df, selected_time, figsize=(14, 10)
        )
        plt.tight_layout()
        plt.show()


@app.cell
def _(metrics_df, mo, MetricNames):
    metric_choices = [
        MetricNames.total_cells,
        MetricNames.num_genotypes,
        MetricNames.shannon_index,
        MetricNames.simpson_index,
        MetricNames.max_mutations,
    ]

    metric_dropdown = (
        mo.ui.dropdown(
            options=metric_choices,
            value="TotalCells",
            label="Select metric for temporal trend:",
        )
        if metrics_df is not None
        else None
    )
    return (metric_dropdown,)


@app.cell
def _(mo, metric_dropdown):
    mo.md("""Select metric:""")

    mo.hstack([metric_dropdown])
    return


@app.cell
def _(MCVisualization, metric_dropdown, metrics_df, plt):
    if metric_dropdown is None or metrics_df is None:
        pass
    else:
        selected_metric = metric_dropdown.value
        _ax_temporal = MCVisualization.plot_temporal_trend(metrics_df, selected_metric)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app.run()

# TODO: add way to globally scale axis, as changing axis scale at different timepoints makes it hard to compare distributions across time.
