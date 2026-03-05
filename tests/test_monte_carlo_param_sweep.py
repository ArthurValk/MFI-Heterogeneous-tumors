"""marimo edit tests/test_monte_carlo_param_sweep.py"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo
    import polars as pl

    # Add project root to path for imports
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.monte_carlo.monte_carlo import MonteCarloEngine
    from non_spatial.parametrization import ModelParameters, MetricNames
    from tests.test_output import TEST_OUTPUT_PATH

    return (
        MetricNames,
        ModelParameters,
        MonteCarloEngine,
        TEST_OUTPUT_PATH,
        marimo,
        pl,
    )


@app.cell
def _(ModelParameters, TEST_OUTPUT_PATH, marimo):
    import numpy as np

    # Define parameter sweep configuration
    marimo.md("## Parameter Sweep Configuration")

    output_dir = TEST_OUTPUT_PATH / "param_sweep"

    # Base parameters (fixed for all sweep combinations)
    base_params = ModelParameters(
        number_of_genes=50,
        carrying_capacity=1000,
        number_of_generations=1000,
        mutation_rate_per_gene=10**-2,
        fusion_rate=10**-2,
        growth_rate=0.5,
        death_rate=0.5,
        save_path=TEST_OUTPUT_PATH,
        treatment_every=20,
        treatment_duration=20,
        treatment_base_extra_death=0.3,
        treatment_selection=0.1,
        treatment_cell_density_dependence=0.0,
    )

    # Define parameter sweep ranges (5 dimensions × 3 values = 243 combinations)
    sweep_params = {
        "mutation_rate_per_gene": np.array([1e-4, 1e-3, 1e-2]),
        "fusion_rate": np.array([1e-4, 1e-3, 1e-2]),
        "treatment_selection": np.array([0.001, 0.01, 0.1]),
        "treatment_every": np.array([5, 20, 200]),
        "treatment_cell_density_dependence": np.array([0.0, 1.0, 10.0]),
    }

    marimo.md(f"**Fixed Parameters:**")
    marimo.md(f"- Genes: {base_params.number_of_genes}")
    marimo.md(f"- Generations: {base_params.number_of_generations}")
    marimo.md(f"- Carrying Capacity: {base_params.carrying_capacity}")
    marimo.md(f"- Growth Rate = Death Rate: {base_params.growth_rate}")
    marimo.md(f"- Treatment Base Extra Death: {base_params.treatment_base_extra_death}")

    marimo.md(f"**Sweep Parameters (5 dimensions × 3 values each):**")
    num_values_per_param = 3
    for param_name, values in sweep_params.items():
        marimo.md(f"- {param_name}: {list(values)}")

    num_combinations = num_values_per_param ** len(sweep_params)
    marimo.md(f"**Total Combinations:** {num_combinations}")
    marimo.md(f"**Seeds per Combination:** 300")
    marimo.md(
        f"**Total Runs:** {num_combinations * 300} (est. 10.7 hours at 0.5 sec/run)"
    )
    return base_params, output_dir, sweep_params


@app.cell
def _(MonteCarloEngine, base_params, marimo, output_dir, sweep_params):
    # Run parameter sweep using dedicated sweep method
    marimo.md("## Running Parameter Sweep")

    seeds = list(range(300))  # 300 seeds for qualitative distribution assessment

    marimo.md(f"**Starting sweep with {len(seeds)} seeds...**")

    try:
        MonteCarloEngine.monte_carlo_parameter_sweep(
            parameters=base_params,
            seeds=seeds,
            sweep_params=sweep_params,
            save_path=output_dir,
            batch_size=10,  # Smaller batch size for memory efficiency
        )
        marimo.md("✅ Parameter sweep completed!")
        sweep_status = "✅ Completed"
    except Exception as e:
        marimo.md(f"❌ Sweep failed: {e}")
        sweep_status = f"❌ Error: {str(e)}"
    return


@app.cell
def _(marimo, output_dir, pl):
    import json

    marimo.md("## Sweep Metadata and Output Summary")

    # Check if sweep_metadata.json exists
    metadata_file = output_dir / "sweep_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

        marimo.md(f"**Total Combinations:** {metadata['num_combinations']}")
        marimo.md(f"**Seeds per Combination:** {metadata['num_seeds']}")
        marimo.md(f"**Total Runs:** {metadata['total_runs']}")

        # Parse results metadata
        results = metadata.get("results", [])
        summary_data = []
        for result in results:
            combo = result["combination"]
            combo_name = ", ".join(f"{k}={v:.2e}" for k, v in combo.items())
            summary_data.append(
                {
                    "Parameters": combo_name,
                    "Output Dir": result["output_dir"],
                }
            )

        if summary_data:
            summary_df = pl.DataFrame(summary_data)
            marimo.md(f"**Parameter Combinations:**")
            summary_df
        else:
            marimo.md("No results found in metadata")
    else:
        marimo.md("No sweep_metadata.json found")
    return


@app.cell
def _(MetricNames, marimo, output_dir, pl):
    marimo.md("## Metrics Comparison Across Parameter Combinations")

    sweep_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name != ".temp_batches"
    ]
    all_metrics = {}

    for sweep_dir in sorted(sweep_dirs):
        metrics_file = sweep_dir / "metrics_data.parquet"
        if metrics_file.exists():
            df = pl.read_parquet(metrics_file)
            all_metrics[sweep_dir.name] = df

    if all_metrics:
        marimo.md(f"**Loaded metrics from {len(all_metrics)} configurations**")

        # Show summary statistics for each config
        for metrics_combo_name, metrics_df in all_metrics.items():
            marimo.md(f"### {metrics_combo_name}")
            marimo.md(f"- Data points: {metrics_df.shape[0]}")
            marimo.md(f"- Columns: {', '.join(metrics_df.columns)}")
            if MetricNames.total_cells in metrics_df.columns:
                marimo.md(
                    f"- Max population: {metrics_df[MetricNames.total_cells].max():.0f}"
                )
            if MetricNames.num_genotypes in metrics_df.columns:
                marimo.md(
                    f"- Max genotypes: {metrics_df[MetricNames.num_genotypes].max():.0f}"
                )
            if MetricNames.shannon_index in metrics_df.columns:
                marimo.md(
                    f"- Max Shannon: {metrics_df[MetricNames.shannon_index].max():.3f}"
                )
    else:
        marimo.md("No metrics data found")
    return (all_metrics,)


@app.cell
def _(MetricNames, all_metrics, marimo, pl):
    marimo.md("## Final State Comparison Across Combinations")

    if all_metrics:
        final_states = []
        for final_config_name, config_df in all_metrics.items():
            if len(config_df) > 0:
                # Group by seeds and get final value for each seed, then average
                if MetricNames.seed in config_df.columns:
                    final_per_seed = config_df.group_by(MetricNames.seed).agg(
                        [
                            pl.col(MetricNames.time).max(),
                            pl.col(MetricNames.total_cells).last(),
                            pl.col(MetricNames.num_genotypes).last(),
                            pl.col(MetricNames.shannon_index).last(),
                            pl.col(MetricNames.simpson_index).last(),
                            pl.col(MetricNames.max_mutations).last(),
                        ]
                    )
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

                    final_states.append(
                        {
                            "Configuration": final_config_name,
                            "Avg Final Population": f"{avg_stats[1]:.0f}",
                            "Avg Genotypes": f"{avg_stats[2]:.0f}",
                            "Avg Shannon": f"{avg_stats[3]:.3f}",
                            "Avg Simpson": f"{avg_stats[4]:.3f}",
                            "Avg Max Mutations": f"{avg_stats[5]:.0f}",
                        }
                    )

        if final_states:
            comparison_df = pl.DataFrame(final_states)
            marimo.md(f"**Summary Statistics (averages across seeds):**")
            comparison_df
        else:
            marimo.md("No final states found")
    else:
        marimo.md("No metrics data available")
    return


if __name__ == "__main__":
    app.run()
