from pathlib import Path
from dataclasses import replace

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from non_spatial.NonSpatialFusion import ModelRun
from non_spatial.parametrization import ModelParameters, MetricNames
from non_spatial.monte_carlo.visualization import MCVisualization


def plot_stacked_groups(
    lineage: pl.DataFrame,
    group_col: str,
    top_n: int | None = None,
    title: str = "Stacked population",
):
    grouped = (
        lineage.group_by([MetricNames.time, group_col])
        .agg(pl.col(MetricNames.cell_count).sum())
        .sort([MetricNames.time, group_col])
    )

    if len(grouped) == 0:
        print(f"No lineage data to plot for {group_col}.")
        return

    if top_n is not None:
        top_groups = (
            grouped.group_by(group_col)
            .agg(pl.col(MetricNames.cell_count).sum().alias("TotalAcrossTime"))
            .sort("TotalAcrossTime", descending=True)
            .head(top_n)[group_col]
            .to_list()
        )
        grouped = grouped.filter(pl.col(group_col).is_in(top_groups))

    d = grouped.to_dict(as_series=False)
    times_list = d[MetricNames.time]
    groups_list = d[group_col]
    counts_list = d[MetricNames.cell_count]

    lookup = dict(zip(zip(times_list, groups_list), counts_list))

    times = sorted(set(times_list))
    group_ids = sorted(set(groups_list))

    stackplot_data = []
    for g in group_ids:
        stackplot_data.append([lookup.get((t, g), 0) for t in times])

    plt.figure(figsize=(14, 8))
    if len(stackplot_data) > 0:
        plt.stackplot(times, *stackplot_data, alpha=0.9)

    suffix = "" if top_n is None else f" (top {top_n})"
    plt.title(f"{title}{suffix}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Cell count")
    plt.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_monte_carlo_mean_ci(
    metrics: pl.DataFrame,
    metric: str,
    confidence: float = 0.95,
):
    required = {MetricNames.time, MetricNames.seed, metric}
    missing = required - set(metrics.columns)
    if missing:
        print(f"Skipping {metric}; missing columns: {missing}")
        return

    z = 1.96 if confidence == 0.95 else 1.96

    summary = (
        metrics.filter(pl.col(metric).is_finite())
        .group_by(MetricNames.time)
        .agg(
            [
                pl.col(metric).mean().alias("mean"),
                pl.col(metric).std(ddof=1).alias("std"),
                pl.col(metric).count().alias("n"),
            ]
        )
        .sort(MetricNames.time)
        .with_columns(
            [
                pl.when(pl.col("n") > 1)
                .then(pl.col("std") / pl.col("n").sqrt())
                .otherwise(0.0)
                .alias("sem")
            ]
        )
        .with_columns(
            [
                (pl.col("mean") - z * pl.col("sem")).alias("lower"),
                (pl.col("mean") + z * pl.col("sem")).alias("upper"),
            ]
        )
    )

    x = summary[MetricNames.time].to_numpy()
    y = summary["mean"].to_numpy()
    lower = summary["lower"].to_numpy()
    upper = summary["upper"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2, label="Mean")
    plt.fill_between(x, lower, upper, alpha=0.25, label=f"{int(confidence * 100)}% CI")
    plt.xlabel("Time (hours)")
    plt.ylabel(metric)
    plt.title(f"{metric}: Monte Carlo mean ± {int(confidence * 100)}% CI")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_treatment_series(metrics: pl.DataFrame):
    needed = [
        MetricNames.time,
        MetricNames.drug_concentration,
        MetricNames.drug_extra_death_wt,
    ]
    if not all(col in metrics.columns for col in needed):
        print("Treatment columns not found; skipping treatment plot.")
        return

    x = metrics[MetricNames.time].to_numpy()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        x,
        metrics[MetricNames.drug_concentration].to_numpy(),
        linewidth=2,
        label=MetricNames.drug_concentration,
    )
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Drug concentration")
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        metrics[MetricNames.drug_extra_death_wt].to_numpy(),
        linewidth=2,
        label=MetricNames.drug_extra_death_wt,
    )
    ax2.set_ylabel("WT treatment pressure")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title("Treatment time series")
    plt.tight_layout()
    plt.show()


def plot_genotype_heatmap(genotypes: pl.DataFrame, max_genotypes: int = 100):
    data = genotypes.filter(pl.col("GenotypeId") < max_genotypes)
    if len(data) == 0:
        print("No genotype data to plot.")
        return

    locus_columns = [col for col in data.columns if col.startswith("Locus")]
    if not locus_columns:
        print("No locus columns found.")
        return

    genotype_ids = data["GenotypeId"].to_list()
    locus_data = data.select(locus_columns).cast(pl.Int32).to_numpy()

    num_genotypes = len(genotype_ids)
    num_loci = len(locus_columns)
    figsize = (max(12, num_loci / 3), max(6, num_genotypes / 2))

    plt.figure(figsize=figsize)
    plt.imshow(
        locus_data,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    plt.xlabel("Locus")
    plt.ylabel("Genotype ID")
    plt.title(f"Genotype mutation patterns (first {max_genotypes})")
    plt.tight_layout()
    plt.show()


def plot_final_population_histogram(metrics: pl.DataFrame):
    # find last recorded time
    last_time = metrics[MetricNames.time].max()

    final_counts = metrics.filter(pl.col(MetricNames.time) == last_time).select(
        [MetricNames.seed, MetricNames.total_cells]
    )

    counts = final_counts[MetricNames.total_cells].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=20)
    plt.xlabel("Total population size")
    plt.ylabel("Frequency")
    plt.title(f"Final population size distribution (t = {last_time})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    n_runs = 100
    preview_every = 10

    base_params = ModelParameters(
        number_of_genes=100,
        carrying_capacity=100000,
        number_of_generations=24 * 4 * 140,  # 140 days
        mutation_rate_per_gene=1e-4,
        fusion_rate=1.4e-3,
        growth_rate=0.12 / 48.0,
        death_rate=0.04 / 48.0,
        save_path=Path("Results") / "DecayTreatmentDemo_MC",
        dt=0.25,
        data_resolution=4,
        diversity=1,
        initial_population_size=1000,
        seed=0,
        treatment_injection_every=21 * 24 * 4,
        treatment_initial_concentration=0.25,
        treatment_halflife=12.0,
        treatment_concentration_to_extra_death=0.7,
        treatment_selection=0.1,
        treatment_resistivity=1.0,
    )

    all_metrics = []

    for run_idx in range(n_runs):
        seed = run_idx
        params = replace(base_params, seed=seed)

        print(f"Running simulation {run_idx + 1}/{n_runs} (seed={seed})...")
        result = ModelRun(params)

        metrics = pl.read_csv(result.metrics_path).with_columns(
            pl.lit(seed).alias(MetricNames.seed)
        )
        all_metrics.append(metrics)

        if (run_idx + 1) % preview_every == 0:
            lineage = pl.read_csv(result.lineage_path)

            print(
                f"Previewing lineage plots for simulation {run_idx + 1} (seed={seed})"
            )

            # per strand / ancestor
            plot_stacked_groups(
                lineage,
                group_col=MetricNames.ancestor_id,
                top_n=None,
                title=f"Seed {seed}: cell counts per strand (all strands)",
            )

            # per genotype, also without top-30 filtering
            plot_stacked_groups(
                lineage,
                group_col=MetricNames.genotype_id,
                top_n=None,
                title=f"Seed {seed}: cell counts per genotype (all genotypes)",
            )

    metrics_all = pl.concat(all_metrics, how="vertical_relaxed")

    out_dir = Path("Results") / "DecayTreatmentDemo_MC"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_all.write_csv(out_dir / "combined_metrics_100_runs.csv")

    print("Saved combined metrics to:", out_dir / "combined_metrics_100_runs.csv")

    # Monte Carlo mean + confidence interval plots
    for metric in [
        MetricNames.total_cells,
        MetricNames.num_genotypes,
        MetricNames.max_mutations,
        MetricNames.shannon_index,
        MetricNames.simpson_index,
    ]:
        plot_monte_carlo_mean_ci(metrics_all, metric, confidence=0.95)

    # If you still want the old percentile-band MCVisualization plots, keep these too:
    for metric in [
        MetricNames.total_cells,
        MetricNames.num_genotypes,
        MetricNames.max_mutations,
        MetricNames.shannon_index,
        MetricNames.simpson_index,
    ]:
        MCVisualization.plot_temporal_trend(metrics_all, metric)
        plt.tight_layout()
        plt.show()

    # Distribution plots at representative times across all 100 runs
    available_times = sorted(metrics_all[MetricNames.time].unique().to_list())
    if len(available_times) >= 3:
        chosen_times = [
            available_times[0],
            available_times[len(available_times) // 2],
            available_times[-1],
        ]
        for t in chosen_times:
            metrics_to_plot = [
                (MetricNames.total_cells, "float"),
                (MetricNames.num_genotypes, "integer"),
                (MetricNames.max_mutations, "integer"),
                (MetricNames.shannon_index, "float"),
                (MetricNames.simpson_index, "float"),
            ]

            # only include treatment metrics if they exist
            if MetricNames.drug_concentration in metrics_all.columns:
                metrics_to_plot.append((MetricNames.drug_concentration, "float"))
            if MetricNames.drug_extra_death_wt in metrics_all.columns:
                metrics_to_plot.append((MetricNames.drug_extra_death_wt, "float"))

            MCVisualization.plot_metric_distributions_at_time(
                metrics_all,
                t,
                metrics_to_plot=metrics_to_plot,
                figsize=(14, 12),
            )
            plt.tight_layout()
            plt.show()

    # Optional: inspect genotype heatmap from the last run
    genotypes = pl.read_csv(result.genotype_path)
    plot_genotype_heatmap(genotypes, max_genotypes=100)
    plot_final_population_histogram(metrics_all)


if __name__ == "__main__":
    main()
