"""marimo edit tests/test_monte_carlo_param_sweep_analysis.py"""

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
    from tests.test_output import TEST_OUTPUT_PATH
    import polars as pl

    return MetricNames, TEST_OUTPUT_PATH, mo, pl, plt


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

    # Create sliders for each parameter dynamically
    param_sliders = []
    for _param_name, _param_values in sorted(param_ranges.items()):
        _slider = mo.ui.slider(steps=_param_values, label=_param_name)
        param_sliders.append(_slider)
    return param_ranges, param_sliders


@app.cell
def _(mo, param_sliders):
    mo.md("""Select parameter combination:""")
    mo.vstack(param_sliders)
    return


@app.cell
def _(output_dir, param_ranges, param_sliders, pl):
    # Get selected values by zipping sorted param names with sliders
    selected_params = {}
    for (_param_name, _param_values), _slider in zip(
        sorted(param_ranges.items()), param_sliders
    ):
        selected_params[_param_name] = _slider.value

    # Lazily scan all metrics files and filter by selected parameters
    from non_spatial.monte_carlo.monte_carlo import _load_results_lazily

    _lazy_metrics = _load_results_lazily(save_path=output_dir, data_source="metrics")

    for _param_name, _param_val in selected_params.items():
        _filtered_lazy = _lazy_metrics.filter(pl.col(_param_name) == _param_val)

    filtered_metrics = _lazy_metrics.collect()

    if len(filtered_metrics) == 0:
        filtered_metrics = None

    return (filtered_metrics,)


@app.cell
def _(filtered_metrics, mo):
    from non_spatial.monte_carlo.visualization import MCVisualization

    mo.md("## Filtered Metrics Data")

    _status = (
        mo.md(
            f"**Loaded {filtered_metrics.shape[0]} rows**\n\n**Columns:** {', '.join(filtered_metrics.columns)}"
        )
        if filtered_metrics is not None
        else mo.md("❌ No filtered metrics available")
    )
    _status
    return (MCVisualization,)


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
def _(MCVisualization, MetricNames, filtered_metrics, mo, plt):
    mo.md("## Filtered Metrics Visualization")

    if filtered_metrics is not None and len(filtered_metrics) > 0:
        mo.md("### Temporal Trends")
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
        plt.show()

    _output = (
        None
        if filtered_metrics is not None and len(filtered_metrics) > 0
        else mo.md(
            "❌ No filtered metrics available - select a parameter combination above"
        )
    )
    _output
    return


@app.cell
def _(mo, time_slider_filtered):
    mo.md("""Select time point for empirical distributions:""")
    mo.hstack([time_slider_filtered])
    return


@app.cell
def _(
    MCVisualization,
    MetricNames,
    filtered_metrics,
    mo,
    time_slider_filtered,
):
    # Empirical distributions at selected time:
    mo.md("### Metric Distributions at Selected Time")
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


@app.cell
def _(mo, output_dir, pl):
    mo.md("## Load All Metrics Data")
    mo.md("*(For comparison across all parameter combinations)*")

    all_metrics = {}
    sweep_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    for _sweep_dir in sorted(sweep_dirs):
        _metrics_path = _sweep_dir / "metrics_data.parquet"
        if _metrics_path.exists():
            _df = pl.read_parquet(_metrics_path)
            all_metrics[_sweep_dir.name] = _df

    mo.md(f"**Loaded metrics from {len(all_metrics)} parameter combinations**")
    return (all_metrics,)


@app.cell
def _(MetricNames, all_metrics, mo, pl):
    mo.md("## Summary Statistics by Parameter Combination")

    summary_rows = []
    for _combo_name, _df in sorted(all_metrics.items()):
        if len(_df) > 0 and MetricNames.seed in _df.columns:
            # Group by seed, get final values, then average
            final_per_seed = _df.group_by(MetricNames.seed).agg(
                [
                    pl.col(MetricNames.time).max(),
                    pl.col(MetricNames.total_cells).last(),
                    pl.col(MetricNames.num_genotypes).last(),
                    pl.col(MetricNames.shannon_index).last(),
                    pl.col(MetricNames.simpson_index).last(),
                    pl.col(MetricNames.max_mutations).last(),
                ]
            )

            if len(final_per_seed) > 0:
                avg_stats = final_per_seed.select(
                    [
                        pl.col(MetricNames.time).mean(),
                        pl.col(MetricNames.total_cells).mean(),
                        pl.col(MetricNames.num_genotypes).mean(),
                        pl.col(MetricNames.shannon_index).mean(),
                        pl.col(MetricNames.simpson_index).mean(),
                        pl.col(MetricNames.max_mutations).mean(),
                    ]
                ).row(0)

                summary_rows.append(
                    {
                        "Parameters": _combo_name,
                        "Avg Population": f"{avg_stats[1]:.0f}",
                        "Avg Genotypes": f"{avg_stats[2]:.1f}",
                        "Avg Shannon": f"{avg_stats[3]:.3f}",
                        "Avg Simpson": f"{avg_stats[4]:.3f}",
                        "Avg Max Mutations": f"{avg_stats[5]:.1f}",
                    }
                )

    summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    summary_df
    return


@app.cell
def _(metadata, mo):
    import itertools

    mo.md("## Filter Parameter Combinations for Comparison")

    # Reconstruct combinations from metadata
    param_grids = metadata.get("param_grids", {})
    param_names = sorted(param_grids.keys())
    param_values_lists = [param_grids[name] for name in param_names]
    _all_combinations = list(itertools.product(*param_values_lists))

    # Format option labels
    _combo_labels = []
    for _combo_vals in _all_combinations:
        _label = ", ".join(
            f"{name}={val:.4g}" for name, val in zip(param_names, _combo_vals)
        )
        _combo_labels.append(_label)

    # Create multiselect with first combination selected by default
    combo_multiselect = mo.ui.multiselect(
        options=_combo_labels,
        value=[_combo_labels[0]] if _combo_labels else [],
        label="Select parameter combinations to compare:",
    )
    return (combo_multiselect,)


@app.cell
def _(combo_multiselect, mo):
    mo.md("""Select which combinations to plot:""")
    mo.vstack([combo_multiselect])
    return


@app.cell
def _(MCVisualization, MetricNames, all_metrics, combo_multiselect, mo, plt):
    mo.md("## Population Dynamics Across All Combinations")

    # Filter all_metrics by selected combinations
    _selected_combos = combo_multiselect.value if combo_multiselect.value else []
    _filtered_metrics = {
        _name: _df
        for _name, _df in all_metrics.items()
        if any(_combo in _name for _combo in _selected_combos)
    }

    if (
        _filtered_metrics
        and MetricNames.total_cells
        in _filtered_metrics[list(_filtered_metrics.keys())[0]].columns
    ):
        _dfs = list(_filtered_metrics.values())
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
        plt.show()
    return


@app.cell
def _(MetricNames, all_metrics):
    """Compute available times from all metrics."""
    _available_times = set()
    for _df in all_metrics.values():
        if MetricNames.time in _df.columns:
            _available_times.update(_df[MetricNames.time].unique().to_list())

    available_times = sorted(list(_available_times))
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

    mo.hstack([time_slider])
    return


@app.cell
def _(
    MCVisualization,
    MetricNames,
    all_metrics,
    combo_multiselect,
    mo,
    plt,
    time_slider,
):
    mo.md("## Final State Distributions")

    _selected_combos = combo_multiselect.value if combo_multiselect.value else []
    _filtered_metrics = {
        _name: _df
        for _name, _df in all_metrics.items()
        if any(_combo in _name for _combo in _selected_combos)
    }
    if _filtered_metrics and time_slider.value is not None:
        _selected_time = time_slider.value
        _metrics_to_plot = [
            (MetricNames.total_cells, "float"),
            (MetricNames.num_genotypes, "integer"),
            (MetricNames.shannon_index, "float"),
            (MetricNames.max_mutations, "integer"),
        ]

        # Show distributions for each parameter combination at selected time
        for _combo_name, _df in sorted(_filtered_metrics.items()):
            if MetricNames.time in _df.columns:
                _fig = MCVisualization.plot_metric_distributions_at_time(
                    _df,
                    _selected_time,
                    metrics_to_plot=_metrics_to_plot,
                    figsize=(12, 8),
                )
                plt.show()
    return


@app.cell
def _(MetricNames, all_metrics, mo, pl):
    mo.md("## Extinction Risk Analysis")

    _extinction_analysis = []

    for _combo_name, _df in sorted(all_metrics.items()):
        if len(_df) > 0 and MetricNames.seed in _df.columns:
            # Get final population for each seed
            _final_per_seed = _df.group_by(MetricNames.seed).agg(
                pl.col(MetricNames.total_cells).last()
            )

            # Count extinctions (population = 0)
            _extinct_count = _final_per_seed.filter(
                pl.col(MetricNames.total_cells) == 0
            )
            _extinctions = len(_extinct_count)
            _total_seeds = len(_final_per_seed)
            _extinction_rate = (
                _extinctions / _total_seeds * 100 if _total_seeds > 0 else 0
            )

            _extinction_analysis.append(
                {
                    "Parameters": _combo_name,
                    "Extinctions": _extinctions,
                    "Total Seeds": int(_total_seeds),
                    "Extinction Rate (%)": f"{_extinction_rate:.1f}",
                }
            )

    _extinction_df = (
        pl.DataFrame(_extinction_analysis) if _extinction_analysis else pl.DataFrame()
    )

    if _extinction_analysis:
        mo.md("**Tumor extinction rates across parameter combinations:**")

    _extinction_df
    return


if __name__ == "__main__":
    app.run()
