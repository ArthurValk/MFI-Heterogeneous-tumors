from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl

from non_spatial.parametrization import ModelParameters, MetricNames
from non_spatial.monte_carlo.monte_carlo import MonteCarloEngine


def summarize_temporal(metrics_df: pl.DataFrame):
    return (
        metrics_df.sort(MetricNames.time)
        .group_by(MetricNames.time)
        .agg(
            [
                pl.col(MetricNames.total_cells).mean().alias("mean"),
                pl.col(MetricNames.total_cells).quantile(0.05).alias("lower"),
                pl.col(MetricNames.total_cells).quantile(0.95).alias("upper"),
            ]
        )
        .sort(MetricNames.time)
    )


def plot_time_series_with_bounds(metrics_on, metrics_off, output_dir):

    on_summary = summarize_temporal(metrics_on)
    off_summary = summarize_temporal(metrics_off)

    fig, ax = plt.subplots(figsize=(11, 6))

    x_on = on_summary[MetricNames.time].to_numpy()
    mean_on = on_summary["mean"].to_numpy()
    low_on = on_summary["lower"].to_numpy()
    up_on = on_summary["upper"].to_numpy()

    x_off = off_summary[MetricNames.time].to_numpy()
    mean_off = off_summary["mean"].to_numpy()
    low_off = off_summary["lower"].to_numpy()
    up_off = off_summary["upper"].to_numpy()

    ax.fill_between(x_on, low_on, up_on, color="blue", alpha=0.25)
    ax.plot(x_on, mean_on, color="blue", linewidth=2.5, label="Fusion ON")

    ax.fill_between(x_off, low_off, up_off, color="red", alpha=0.25)
    ax.plot(x_off, mean_off, color="red", linewidth=2.5, label="Fusion OFF")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Total cells")
    ax.set_title("Population size with 5–95% confidence bounds")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "cell_count_time_series_bounds.png", dpi=300)
    plt.close()


def plot_final_histograms(metrics_on, metrics_off, output_dir):

    final_time = min(
        metrics_on[MetricNames.time].max(),
        metrics_off[MetricNames.time].max(),
    )

    on_final = metrics_on.filter(pl.col(MetricNames.time) == final_time)
    off_final = metrics_off.filter(pl.col(MetricNames.time) == final_time)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        on_final[MetricNames.total_cells].to_numpy(),
        bins=25,
        density=True,
        alpha=0.5,
        color="blue",
        label="Fusion ON",
    )

    ax.hist(
        off_final[MetricNames.total_cells].to_numpy(),
        bins=25,
        density=True,
        alpha=0.5,
        color="red",
        label="Fusion OFF",
    )

    ax.set_xlabel("Final population size")
    ax.set_ylabel("Density")
    ax.set_title(f"Final-time cell count distribution (t={final_time}h)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "final_population_histograms.png", dpi=300)
    plt.close()


def preview_trajectories(metrics_df, color, label, ax, preview_every=10):

    seeds = sorted(metrics_df[MetricNames.seed].unique().to_list())

    for s in seeds[::preview_every]:
        df = metrics_df.filter(pl.col(MetricNames.seed) == s).sort(MetricNames.time)

        ax.plot(
            df[MetricNames.time].to_numpy(),
            df[MetricNames.total_cells].to_numpy(),
            color=color,
            alpha=0.15,
            linewidth=1,
        )

    summary = summarize_temporal(metrics_df)

    ax.plot(
        summary[MetricNames.time].to_numpy(),
        summary["mean"].to_numpy(),
        color=color,
        linewidth=2.5,
        label=label,
    )


def plot_preview(metrics_on, metrics_off, output_dir):

    fig, ax = plt.subplots(figsize=(11, 6))

    preview_trajectories(metrics_on, "blue", "Fusion ON mean", ax)
    preview_trajectories(metrics_off, "red", "Fusion OFF mean", ax)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Total cells")
    ax.set_title("Preview of Monte Carlo trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_preview.png", dpi=300)
    plt.close()


def main():

    n_runs = 100
    preview_every = 10

    base_params = ModelParameters(
        number_of_genes=100,
        carrying_capacity=100000,
        number_of_generations=24 * 4 * 140,
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

    seeds = list(range(n_runs))

    root = base_params.save_path
    fusion_on_dir = root / "fusion_on"
    fusion_off_dir = root / "fusion_off"
    plot_dir = root / "plots"

    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Running simulations WITH fusion")
    MonteCarloEngine.monte_carlo_simulation(
        parameters=base_params,
        seeds=seeds,
        save_path=fusion_on_dir,
        batch_size=n_runs,
    )

    print("Running simulations WITHOUT fusion")

    no_fusion_params = ModelParameters(
        **{**base_params.__dict__, "fusion_rate": 0.0}
    )

    MonteCarloEngine.monte_carlo_simulation(
        parameters=no_fusion_params,
        seeds=seeds,
        save_path=fusion_off_dir,
        batch_size=n_runs,
    )

    metrics_on = pl.read_parquet(fusion_on_dir / "metrics_data.parquet")
    metrics_off = pl.read_parquet(fusion_off_dir / "metrics_data.parquet")

    plot_final_histograms(metrics_on, metrics_off, plot_dir)

    plot_time_series_with_bounds(metrics_on, metrics_off, plot_dir)

    plot_preview(metrics_on, metrics_off, plot_dir)

    print("Done.")
    print("Plots saved to:", plot_dir)


if __name__ == "__main__":
    main()