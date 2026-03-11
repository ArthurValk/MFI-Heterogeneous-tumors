"""Visualization utilities for Monte Carlo simulation results."""

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from scipy import stats

from non_spatial.parametrization import MetricNames


class MCVisualization:
    """Helper functions for visualizing Monte Carlo simulation results."""

    @staticmethod
    def load_simulation_results(parquet_dir: Path) -> dict[str, pl.DataFrame]:
        parquet_dir = Path(parquet_dir)
        results = {}

        lineage_path = parquet_dir / "lineage_data.parquet"
        if lineage_path.exists():
            results["lineage"] = pl.read_parquet(lineage_path)

        metrics_path = parquet_dir / "metrics_data.parquet"
        if metrics_path.exists():
            results["metrics"] = pl.read_parquet(metrics_path)

        genotypes_path = parquet_dir / "genotypes_data.parquet"
        if genotypes_path.exists():
            results["genotypes"] = pl.read_parquet(genotypes_path)

        return results

    @staticmethod
    def get_available_times(metrics_df: pl.DataFrame) -> list[int | float]:
        return sorted(metrics_df[MetricNames.time].unique().to_list())

    @staticmethod
    def get_metrics_at_time(metrics_df: pl.DataFrame, time: int | float) -> pl.DataFrame:
        return metrics_df.filter(pl.col(MetricNames.time) == time)

    @staticmethod
    def plot_empirical_pdf(
        data: np.ndarray,
        title: str = "",
        bins: int = 30,
        ax: Optional[matplotlib.axes.Axes] = None,
        integer_valued: bool = False,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        clean_data = data[np.isfinite(data)]

        if len(clean_data) == 0:
            ax.text(0.5, 0.5, "No finite data available", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            return ax

        if integer_valued:
            min_val = int(np.floor(clean_data.min()))
            max_val = int(np.ceil(clean_data.max()))
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

        ax.hist(
            clean_data,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="black",
        )

        if not integer_valued and len(clean_data) > 1 and clean_data.std() > 0:
            kde = stats.gaussian_kde(clean_data)
            x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
            ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            ax.legend()

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_metric_distributions_at_time(
        metrics_df: pl.DataFrame,
        time: int | float,
        metrics_to_plot: Optional[list[tuple[str, Literal["integer", "float"]]]] = None,
        figsize: tuple = (14, 12),
    ) -> matplotlib.figure.Figure:
        if metrics_to_plot is None:
            metrics_to_plot = [
                (MetricNames.total_cells, "float"),
                (MetricNames.num_genotypes, "integer"),
                (MetricNames.max_mutations, "integer"),
                (MetricNames.shannon_index, "float"),
                (MetricNames.simpson_index, "float"),
                (MetricNames.drug_concentration, "float"),
                (MetricNames.drug_extra_death_wt, "float"),
            ]

        data_at_time = MCVisualization.get_metrics_at_time(metrics_df, time)

        if len(data_at_time) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data at time={time}", ha="center", va="center")
            return fig

        n_plots = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, (metric, value_type) in enumerate(metrics_to_plot):
            if metric not in data_at_time.columns:
                continue

            values = data_at_time[metric].to_numpy()
            is_integer = value_type == "integer"
            MCVisualization.plot_empirical_pdf(
                values,
                title=f"{metric} at t={time}",
                ax=axes[idx],
                integer_valued=is_integer,
            )

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Metric Distributions at Time={time}", fontsize=14, fontweight="bold"
        )
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_temporal_trend(
        metrics_df: pl.DataFrame,
        metric: str,
        seed: Optional[int] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        percentile: float = 5.0,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if seed is not None:
            seed_data = metrics_df.filter(pl.col(MetricNames.seed) == seed).sort(
                MetricNames.time
            )
            times = seed_data[MetricNames.time].to_numpy()
            values = seed_data[metric].to_numpy()
            ax.plot(times, values, linewidth=2, label=f"Seed {seed}")
        else:
            clean_data = metrics_df.filter(pl.col(metric).is_finite())

            grouped = (
                clean_data.sort(MetricNames.time)
                .group_by(MetricNames.time)
                .agg(
                    [
                        pl.col(metric).mean().alias("mean"),
                        pl.col(metric).quantile(percentile / 100.0).alias("lower"),
                        pl.col(metric)
                        .quantile(1.0 - percentile / 100.0)
                        .alias("upper"),
                    ]
                )
                .sort(MetricNames.time)
            )

            times = grouped[MetricNames.time].to_numpy()
            values_mean = grouped["mean"].to_numpy()
            values_lower = grouped["lower"].to_numpy()
            values_upper = grouped["upper"].to_numpy()

            ax.fill_between(
                times,
                values_lower,
                values_upper,
                alpha=0.3,
                color="steelblue",
                label=f"{percentile:.0f}th-{100.0 - percentile:.0f}th percentile",
            )

            ax.plot(times, values_mean, "r-", linewidth=2.5, label="Mean", zorder=10)

        ax.set_xlabel("Time (hours)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax