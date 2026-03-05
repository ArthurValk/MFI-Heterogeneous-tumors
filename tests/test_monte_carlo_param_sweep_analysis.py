"""marimo edit tests/test_monte_carlo_param_sweep_analysis.py"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo

    # Add project root to path for imports
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.parametrization import MetricNames
    from tests.test_output import TEST_OUTPUT_PATH
    import polars as pl

    return MetricNames, TEST_OUTPUT_PATH, marimo, Path, pl


@app.cell
def _(TEST_OUTPUT_PATH, marimo):
    import json

    marimo.md("## Parameter Sweep Analysis")
    marimo.md("Load results from completed parameter sweep")

    output_dir = TEST_OUTPUT_PATH / "param_sweep"
    metadata_file = output_dir / "sweep_metadata.json"

    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        marimo.md(f"✅ Sweep metadata found")
        marimo.md(f"- Combinations: {metadata['num_combinations']}")
        marimo.md(f"- Seeds per combination: {metadata['num_seeds']}")
        marimo.md(f"- Total runs: {metadata['total_runs']}")
    else:
        marimo.md(f"❌ No sweep_metadata.json found at {output_dir}")
        metadata = {}

    return output_dir, metadata


@app.cell
def _(output_dir, marimo, pl):
    marimo.md("## Load All Metrics Data")

    all_metrics = {}
    sweep_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    for sweep_dir in sorted(sweep_dirs):
        metrics_path = sweep_dir / "metrics_data.parquet"
        if metrics_path.exists():
            df = pl.read_parquet(metrics_path)
            all_metrics[sweep_dir.name] = df

    marimo.md(f"**Loaded metrics from {len(all_metrics)} parameter combinations**")

    return all_metrics


@app.cell
def _(marimo, all_metrics, MetricNames, pl):
    marimo.md("## Summary Statistics by Parameter Combination")

    if all_metrics:
        summary_rows = []
        for combo_name, df in sorted(all_metrics.items()):
            if len(df) > 0 and MetricNames.seed in df.columns:
                # Group by seed, get final values, then average
                final_per_seed = df.group_by(MetricNames.seed).agg(
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
                            "Parameters": combo_name,
                            "Avg Population": f"{avg_stats[1]:.0f}",
                            "Avg Genotypes": f"{avg_stats[2]:.1f}",
                            "Avg Shannon": f"{avg_stats[3]:.3f}",
                            "Avg Simpson": f"{avg_stats[4]:.3f}",
                            "Avg Max Mutations": f"{avg_stats[5]:.1f}",
                        }
                    )

        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            marimo.md(f"**{len(summary_df)} parameter combinations analyzed**")
            summary_df
        else:
            marimo.md("No valid data found in metrics")
    else:
        marimo.md("No metrics data loaded")


@app.cell
def _(marimo, all_metrics, MetricNames, pl):
    import matplotlib.pyplot as plt

    marimo.md("## Population Dynamics Across All Combinations")

    if (
        all_metrics
        and MetricNames.total_cells in all_metrics[list(all_metrics.keys())[0]].columns
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Total cells over time
        ax = axes[0, 0]
        for combo_name, df in all_metrics.items():
            if MetricNames.time in df.columns and MetricNames.total_cells in df.columns:
                # Average across seeds for cleaner plot
                avg_by_time = (
                    df.group_by(MetricNames.time)
                    .agg(pl.col(MetricNames.total_cells).mean())
                    .sort(MetricNames.time)
                )
                ax.plot(
                    avg_by_time[MetricNames.time],
                    avg_by_time[MetricNames.total_cells],
                    label=combo_name,
                    alpha=0.6,
                    linewidth=1.5,
                )
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Cells (avg across seeds)")
        ax.set_title("Population Growth Dynamics")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Number of genotypes over time
        ax = axes[0, 1]
        for combo_name, df in all_metrics.items():
            if (
                MetricNames.time in df.columns
                and MetricNames.num_genotypes in df.columns
            ):
                avg_by_time = (
                    df.group_by(MetricNames.time)
                    .agg(pl.col(MetricNames.num_genotypes).mean())
                    .sort(MetricNames.time)
                )
                ax.plot(
                    avg_by_time[MetricNames.time],
                    avg_by_time[MetricNames.num_genotypes],
                    label=combo_name,
                    alpha=0.6,
                    linewidth=1.5,
                )
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Genotypes (avg across seeds)")
        ax.set_title("Genotypic Diversity")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 3: Shannon index over time
        ax = axes[1, 0]
        for combo_name, df in all_metrics.items():
            if (
                MetricNames.time in df.columns
                and MetricNames.shannon_index in df.columns
            ):
                avg_by_time = (
                    df.group_by(MetricNames.time)
                    .agg(pl.col(MetricNames.shannon_index).mean())
                    .sort(MetricNames.time)
                )
                ax.plot(
                    avg_by_time[MetricNames.time],
                    avg_by_time[MetricNames.shannon_index],
                    label=combo_name,
                    alpha=0.6,
                    linewidth=1.5,
                )
        ax.set_xlabel("Time")
        ax.set_ylabel("Shannon Index (avg across seeds)")
        ax.set_title("Shannon Diversity Index")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Max mutations over time
        ax = axes[1, 1]
        for combo_name, df in all_metrics.items():
            if (
                MetricNames.time in df.columns
                and MetricNames.max_mutations in df.columns
            ):
                avg_by_time = (
                    df.group_by(MetricNames.time)
                    .agg(pl.col(MetricNames.max_mutations).mean())
                    .sort(MetricNames.time)
                )
                ax.plot(
                    avg_by_time[MetricNames.time],
                    avg_by_time[MetricNames.max_mutations],
                    label=combo_name,
                    alpha=0.6,
                    linewidth=1.5,
                )
        ax.set_xlabel("Time")
        ax.set_ylabel("Max Mutations (avg across seeds)")
        ax.set_title("Maximum Mutation Load")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig
    else:
        marimo.md("Cannot create plots - required columns not found")


@app.cell
def _(marimo, all_metrics, MetricNames, pl):
    import matplotlib.pyplot as plt
    import numpy as np

    marimo.md("## Final State Distributions")

    if all_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract final values for each seed across all combos
        all_final_pop = []
        all_final_geno = []
        all_final_shannon = []
        all_final_mutations = []
        combo_labels = []

        for combo_name, df in sorted(all_metrics.items()):
            if len(df) > 0 and MetricNames.seed in df.columns:
                final_per_seed = df.group_by(MetricNames.seed).agg(
                    [
                        pl.col(MetricNames.total_cells).last(),
                        pl.col(MetricNames.num_genotypes).last(),
                        pl.col(MetricNames.shannon_index).last(),
                        pl.col(MetricNames.max_mutations).last(),
                    ]
                )

                if len(final_per_seed) > 0:
                    all_final_pop.append(
                        final_per_seed[MetricNames.total_cells].to_list()
                    )
                    all_final_geno.append(
                        final_per_seed[MetricNames.num_genotypes].to_list()
                    )
                    all_final_shannon.append(
                        final_per_seed[MetricNames.shannon_index].to_list()
                    )
                    all_final_mutations.append(
                        final_per_seed[MetricNames.max_mutations].to_list()
                    )
                    combo_labels.append(combo_name[:30])  # Truncate for readability

        # Plot 1: Final population distribution
        ax = axes[0, 0]
        ax.boxplot(all_final_pop, labels=combo_labels, vert=True)
        ax.set_ylabel("Final Population")
        ax.set_title("Final Population Distribution")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Final genotypes distribution
        ax = axes[0, 1]
        ax.boxplot(all_final_geno, labels=combo_labels, vert=True)
        ax.set_ylabel("Number of Genotypes")
        ax.set_title("Final Genotypes Distribution")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Final Shannon distribution
        ax = axes[1, 0]
        ax.boxplot(all_final_shannon, labels=combo_labels, vert=True)
        ax.set_ylabel("Shannon Index")
        ax.set_title("Final Shannon Index Distribution")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Final mutations distribution
        ax = axes[1, 1]
        ax.boxplot(all_final_mutations, labels=combo_labels, vert=True)
        ax.set_ylabel("Max Mutations")
        ax.set_title("Final Max Mutations Distribution")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig
    else:
        marimo.md("No metrics data available")


@app.cell
def _(marimo, all_metrics, MetricNames, pl):
    marimo.md("## Extinction Risk Analysis")

    if all_metrics:
        extinction_analysis = []

        for combo_name, df in sorted(all_metrics.items()):
            if len(df) > 0 and MetricNames.seed in df.columns:
                # Get final population for each seed
                final_per_seed = df.group_by(MetricNames.seed).agg(
                    pl.col(MetricNames.total_cells).last()
                )

                # Count extinctions (population = 0)
                extinctions = (final_per_seed[MetricNames.total_cells] == 0).sum()
                total_seeds = len(final_per_seed)
                extinction_rate = (
                    extinctions / total_seeds * 100 if total_seeds > 0 else 0
                )

                extinction_analysis.append(
                    {
                        "Parameters": combo_name,
                        "Extinctions": int(extinctions),
                        "Total Seeds": int(total_seeds),
                        "Extinction Rate (%)": f"{extinction_rate:.1f}",
                    }
                )

        if extinction_analysis:
            extinction_df = pl.DataFrame(extinction_analysis)
            marimo.md(f"**Tumor extinction rates across parameter combinations:**")
            extinction_df
        else:
            marimo.md("No extinction data found")
    else:
        marimo.md("No metrics data available")


if __name__ == "__main__":
    app.run()
