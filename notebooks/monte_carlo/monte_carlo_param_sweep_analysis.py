"""marimo run notebooks/monte_carlo/monte_carlo_param_sweep_analysis.py

Note: this code is designed to analyze the results of a Monte Carlo parameter sweep performed by `notebooks/monte_carlo_param_sweep.py`.
To visualize the results, set the 'output_dir' to the directory containing the results from the sweep, and run this notebook.

If you have not yet performed the parameter sweep, you can run `notebooks/monte_carlo_param_sweep.py` first to generate the necessary output data for analysis.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt

    # Add project root to path for imports
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.parametrization import MetricNames
    from non_spatial.monte_carlo.visualization import MCVisualization
    from tests.test_output import TEST_OUTPUT_PATH
    import polars as pl

    return MCVisualization, MetricNames, TEST_OUTPUT_PATH, mo, pl, plt


@app.cell
def _(TEST_OUTPUT_PATH, mo):
    output_dir = TEST_OUTPUT_PATH / "param_sweep_3"
    from non_spatial.monte_carlo.monte_carlo import _load_metadata

    metadata = _load_metadata(output_dir)

    mo.md("Loaded metadata")
    return metadata, output_dir


@app.cell
def _(metadata, mo):
    # Get parameter ranges from metadata
    param_ranges = metadata.get("param_grids", {})

    # Create sliders for each parameter dynamically using mo.ui.array for proper reactivity
    param_sliders = mo.ui.array(
        [
            mo.ui.slider(steps=_param_values, label=_param_name)
            for _param_name, _param_values in sorted(param_ranges.items())
        ]
    )
    return param_ranges, param_sliders


# <editor-fold desc="Visualizing with sliders for each param">


@app.cell
def _(mo):
    mo.md("""
    ### Select parameter combination:
    """)
    return


@app.cell
def _(mo, param_sliders):
    mo.vstack(param_sliders)
    return


@app.cell
def _(output_dir):
    # Load lazy metrics once (only re-runs if output_dir changes)
    from non_spatial.monte_carlo.monte_carlo import _load_results_lazily

    lazy_metrics = _load_results_lazily(save_path=output_dir, data_source="metrics")
    return (lazy_metrics,)


@app.cell
def _(lazy_metrics, param_ranges, param_sliders, pl):
    # Get selected values by zipping sorted param names with sliders
    selected_params = {}
    for (_param_name, _param_values), _slider in zip(
        sorted(param_ranges.items()), param_sliders
    ):
        selected_params[_param_name] = _slider.value

    # Apply all filters to lazy metrics (chain filters together)
    filtered_lazy = lazy_metrics
    for _param_name, _param_val in selected_params.items():
        filtered_lazy = filtered_lazy.filter(pl.col(_param_name) == _param_val)

    filtered_metrics = filtered_lazy.collect()

    if len(filtered_metrics) == 0:
        filtered_metrics = None
    return (filtered_metrics,)


@app.cell
def _(filtered_metrics, mo):
    _status_str = "## Filtered Metrics Data\n\n"
    if filtered_metrics is not None:
        _status_str += f"**Loaded {filtered_metrics.shape[0]} rows**\n\n**Columns:** {', '.join(filtered_metrics.columns)}"
    else:
        _status_str += "No filtered metrics available"
    mo.md(_status_str)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Filtered Metrics Visualization
    """)
    return


@app.cell
def _(MCVisualization, MetricNames, filtered_metrics, plt):
    if filtered_metrics is not None and len(filtered_metrics) > 0:
        _metrics_to_viz = [
            MetricNames.total_cells,
            MetricNames.num_genotypes,
            MetricNames.shannon_index,
            MetricNames.max_mutations,
        ]

        _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
        _axes = _axes.flatten()

        for _idx, _metric in enumerate(_metrics_to_viz):
            if _metric in filtered_metrics.columns:
                MCVisualization.plot_temporal_trend(
                    filtered_metrics,
                    _metric,
                    ax=_axes[_idx],
                    percentile=5.0,
                )

        plt.tight_layout()

    _fig
    return


@app.cell
def _(filtered_metrics):
    filtered_metrics
    return


@app.cell
def _(MetricNames, filtered_metrics):
    """Compute available times from filtered metrics."""
    available_times_filtered = (
        sorted(filtered_metrics[MetricNames.time].unique().to_list())
        if (
            filtered_metrics is not None
            and MetricNames.time in filtered_metrics.columns
        )
        else []
    )
    return (available_times_filtered,)


@app.cell
def _(available_times_filtered, mo):
    time_slider_filtered = mo.ui.slider(
        steps=available_times_filtered if available_times_filtered else [],
        label="Select timepoint for empirical distributions:",
        value=available_times_filtered[len(available_times_filtered) // 2],
    )
    return (time_slider_filtered,)


@app.cell
def _(mo):
    mo.md("""
    Select time point for empirical distributions:
    """)
    return


@app.cell
def _(mo, time_slider_filtered):
    mo.hstack([time_slider_filtered])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Metric Distributions at Selected Time
    """)
    return


@app.cell
def _(MCVisualization, MetricNames, filtered_metrics, time_slider_filtered):
    # Empirical distributions at selected time:
    _selected_time = time_slider_filtered.value
    _metrics_to_plot = [
        (MetricNames.total_cells, "float"),
        (MetricNames.num_genotypes, "integer"),
        (MetricNames.shannon_index, "float"),
        (MetricNames.max_mutations, "integer"),
    ]
    _dist_fig = MCVisualization.plot_metric_distributions_at_time(
        filtered_metrics,
        _selected_time,
        metrics_to_plot=_metrics_to_plot,
        figsize=(12, 8),
    )
    _dist_fig
    return


# </editor-fold>


@app.cell
def _(lazy_metrics, mo, param_ranges):
    # Get unique parameter combinations without materializing data
    param_cols = list(sorted(param_ranges.keys()))
    return (lazy_metrics, param_cols)


@app.cell
def _(MetricNames, lazy_metrics, param_cols, pl):
    # Compute summary statistics using lazy evaluation
    # Group by parameters, then by seed, get final values and average
    _by_params_and_seed = lazy_metrics.group_by(param_cols + [MetricNames.seed]).agg(
        [
            pl.col(MetricNames.time).max(),
            pl.col(MetricNames.total_cells).last(),
            pl.col(MetricNames.num_genotypes).last(),
            pl.col(MetricNames.shannon_index).last(),
            pl.col(MetricNames.simpson_index).last(),
            pl.col(MetricNames.max_mutations).last(),
        ]
    )

    # Then average across seeds for each parameter combination
    summary_lazy = _by_params_and_seed.group_by(param_cols).agg(
        [
            pl.col(MetricNames.time).mean(),
            pl.col(MetricNames.total_cells).mean(),
            pl.col(MetricNames.num_genotypes).mean(),
            pl.col(MetricNames.shannon_index).mean(),
            pl.col(MetricNames.simpson_index).mean(),
            pl.col(MetricNames.max_mutations).mean(),
        ]
    )

    # Only collect when displaying
    summary_df = summary_lazy.collect().select(
        param_cols
        + [
            pl.col(MetricNames.total_cells).alias("Avg Population"),
            pl.col(MetricNames.num_genotypes).alias("Avg Genotypes"),
            pl.col(MetricNames.shannon_index).alias("Avg Shannon"),
            pl.col(MetricNames.simpson_index).alias("Avg Simpson"),
            pl.col(MetricNames.max_mutations).alias("Avg Max Mutations"),
        ]
    )
    summary_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Filter Parameter Combinations for Comparison
    """)
    return


@app.cell
def _(metadata, mo):
    import itertools

    # Reconstruct combinations from metadata
    param_grids = metadata.get("param_grids", {})
    param_names = sorted(param_grids.keys())
    param_values_lists = [param_grids[name] for name in param_names]
    _all_combinations = list(itertools.product(*param_values_lists))

    # Format option labels and create mapping to parameter dicts
    _combo_labels = []
    combo_to_params = {}
    for _combo_vals in _all_combinations:
        _label = ", ".join(
            f"{name}={val:.4g}" for name, val in zip(param_names, _combo_vals)
        )
        _combo_labels.append(_label)
        combo_to_params[_label] = dict(zip(param_names, _combo_vals))

    # Create multiselect with first combination selected by default
    combo_multiselect = mo.ui.multiselect(
        options=_combo_labels,
        value=[_combo_labels[0]] if _combo_labels else [],
        label="Select parameter combinations to compare:",
    )
    return (combo_multiselect, combo_to_params)


@app.cell
def _(mo):
    mo.md("""
    ## Plot multiple parameter combinations


    Select which combinations to plot:
    """)
    return


@app.cell
def _(combo_multiselect, mo):
    mo.vstack([combo_multiselect])
    return


@app.cell
def _(
    MCVisualization,
    MetricNames,
    combo_multiselect,
    combo_to_params,
    lazy_metrics,
    pl,
    plt,
):
    # Filter lazy_metrics by selected parameter combinations
    _selected_combos = combo_multiselect.value if combo_multiselect.value else []
    _dfs = []

    if not _selected_combos:
        _fig = None
    else:
        # Collect data for each combo separately to plot as separate lines
        for _combo_label in _selected_combos:
            _param_dict = combo_to_params[_combo_label]
            _combo_cond = None
            for _param, _val in _param_dict.items():
                _param_cond = pl.col(_param) == _val
                _combo_cond = (
                    _param_cond if _combo_cond is None else _combo_cond & _param_cond
                )
            _combo_data = lazy_metrics.filter(_combo_cond).collect()
            if len(_combo_data) > 0:
                _dfs.append(_combo_data)

        if _dfs and MetricNames.total_cells in _dfs[0].columns:
            _metrics_to_plot = [
                MetricNames.total_cells,
                MetricNames.num_genotypes,
                MetricNames.shannon_index,
                MetricNames.max_mutations,
            ]

            _fig, _axes = plt.subplots(2, 2, figsize=(16, 12))
            _axes = _axes.flatten()

            for _idx, _metric in enumerate(_metrics_to_plot):
                MCVisualization.plot_temporal_trend(
                    _dfs,
                    _metric,
                    ax=_axes[_idx],
                    percentile=5.0,
                )

            plt.tight_layout()
        else:
            _fig = None

    _fig
    return


@app.cell
def _(MetricNames, lazy_metrics, pl):
    """Compute available times from lazy metrics using lazy evaluation."""
    if MetricNames.time in lazy_metrics.collect_schema().names():
        available_times = (
            lazy_metrics.select(pl.col(MetricNames.time).unique().sort())
            .collect()[MetricNames.time]
            .to_list()
        )
    else:
        available_times = []
    return (available_times,)


@app.cell
def _(available_times, mo):
    time_slider = mo.ui.slider(
        steps=available_times,
        label="Select timepoint for empirical distributions:",
        value=available_times[len(available_times) // 2],
    )
    return (time_slider,)


@app.cell
def _(mo, time_slider):
    mo.md("""Select time point for empirical distributions:""")

    mo.vstack([time_slider])
    return


@app.cell
def _(
    MCVisualization,
    MetricNames,
    combo_multiselect,
    combo_to_params,
    lazy_metrics,
    mo,
    pl,
    time_slider,
):
    _selected_combos = combo_multiselect.value if combo_multiselect.value else []
    _figs = []

    if _selected_combos and time_slider.value is not None:
        _selected_time = time_slider.value
        _metrics_to_plot = [
            (MetricNames.total_cells, "float"),
            (MetricNames.num_genotypes, "integer"),
            (MetricNames.shannon_index, "float"),
            (MetricNames.max_mutations, "integer"),
        ]

        # Collect distributions for each parameter combination at selected time
        for _combo_label in sorted(_selected_combos):
            _param_dict = combo_to_params[_combo_label]
            # Build filter for this combo
            _combo_cond = None
            for _param, _val in _param_dict.items():
                _param_cond = pl.col(_param) == _val
                _combo_cond = (
                    _param_cond if _combo_cond is None else _combo_cond & _param_cond
                )

            _filtered_df = lazy_metrics.filter(_combo_cond).collect()
            if len(_filtered_df) > 0 and MetricNames.time in _filtered_df.columns:
                _fig = MCVisualization.plot_metric_distributions_at_time(
                    _filtered_df,
                    _selected_time,
                    metrics_to_plot=_metrics_to_plot,
                    figsize=(12, 8),
                )
                _figs.append(_fig)

    # Display all figures stacked vertically
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _(MetricNames, lazy_metrics, param_cols, pl):
    # Compute extinction analysis using lazy evaluation
    # Group by parameters and seed, get final population
    _by_params_and_seed = lazy_metrics.group_by(param_cols + [MetricNames.seed]).agg(
        pl.col(MetricNames.total_cells).last()
    )

    # Count extinctions (population = 0) by parameter combination
    _extinction_lazy = (
        _by_params_and_seed.group_by(param_cols)
        .agg(
            [
                pl.when(pl.col(MetricNames.total_cells) == 0)
                .then(1)
                .otherwise(0)
                .sum()
                .alias("Extinctions"),
                pl.col(MetricNames.seed).count().alias("Total Seeds"),
            ]
        )
        .with_columns(
            (pl.col("Extinctions") / pl.col("Total Seeds") * 100).alias(
                "Extinction Rate (%)"
            )
        )
    )

    # Only collect when displaying
    _extinction_df = _extinction_lazy.collect()
    _extinction_df
    return


if __name__ == "__main__":
    app.run()
