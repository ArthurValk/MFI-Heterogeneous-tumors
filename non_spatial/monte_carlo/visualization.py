"""Visualization utilities for Monte Carlo simulation results."""

from pathlib import Path
from typing import Optional, Literal, Sequence

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec
from scipy import stats

from non_spatial.parametrization import MetricNames

# Standard matplotlib colors for plotting
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


class MCVisualization:
    """Helper functions for visualizing Monte Carlo simulation results."""

    @staticmethod
    def load_simulation_results(parquet_dir: Path) -> dict[str, pl.DataFrame]:
        """Load parquet files from simulation output directory.

        Parameters
        ----------
        parquet_dir : Path
            Directory containing lineage_data.parquet, metrics_data.parquet, etc.

        Returns
        -------
        dict[str, pl.DataFrame]
            Dictionary with keys: 'lineage', 'metrics', 'genotypes'
        """
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
    def get_available_times(metrics_df: pl.DataFrame) -> list[int]:
        """Get sorted list of available timepoints in metrics data.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            Metrics dataframe with MetricNames.time column

        Returns
        -------
        list[int]
            Sorted list of unique timepoints
        """
        return sorted(metrics_df[MetricNames.time].unique().to_list())

    @staticmethod
    def get_metrics_at_time(
        metrics_df: pl.DataFrame, time: int | float
    ) -> pl.DataFrame:
        """Filter metrics dataframe to a specific timepoint.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            Full metrics dataframe
        time : int
            Timepoint to filter to

        Returns
        -------
        pl.DataFrame
            Filtered dataframe with only rows at the specified time
        """
        return metrics_df.filter(pl.col(MetricNames.time) == time)

    @staticmethod
    def plot_empirical_pdf(
        data: np.ndarray,
        title: str = "",
        bins: int = 30,
        ax: Optional[matplotlib.axes.Axes] = None,
        integer_valued: bool = False,
        color_index: int = 0,
    ) -> matplotlib.axes.Axes:
        """Plot empirical probability distribution (histogram + KDE).

        Parameters
        ----------
        data : np.ndarray
            1D array of values
        title : str, optional
            Title for the plot
        bins : int, optional
            Number of histogram bins (default: 30)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new if None)
        integer_valued : bool, optional
            If True, aligns bins to integer boundaries for integer-valued data (default: False)
        color_index : int, optional
            Index into COLORS list for histogram color (default: 0)

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        # Filter out inf and NaN values
        clean_data = data[np.isfinite(data)]

        if len(clean_data) == 0:
            ax.text(0.5, 0.5, "No finite data available", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            return ax

        # Determine bins
        if integer_valued:
            min_val = int(np.floor(clean_data.min()))
            max_val = int(np.ceil(clean_data.max()))
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

        # Histogram
        plot_color = COLORS[color_index % len(COLORS)]
        ax.hist(
            clean_data,
            bins=bins,
            density=True,
            alpha=0.6,
            color=plot_color,
            edgecolor="black",
        )

        # KDE (skip for integer-valued data)
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
        color_index: int = 0,
    ) -> matplotlib.figure.Figure:
        """Plot distributions of multiple metrics at a single timepoint.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            Metrics dataframe
        time : int
            Timepoint to visualize
        metrics_to_plot : list[tuple[str, Literal["integer", "float"]]], optional
            List of (column_name, value_type) tuples to plot. If None, plots standard metrics.
        figsize : tuple, optional
            Figure size (default: (14, 10))
        color_index : int, optional
            Index into COLORS list for histogram color (default: 0)

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing subplots
        """
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

        # Filter to timepoint
        data_at_time = MCVisualization.get_metrics_at_time(metrics_df, time)

        if len(data_at_time) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data at time={time}", ha="center", va="center")
            return fig

        # Create subplots
        n_plots = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # Plot each metric
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
                color_index=color_index,
            )

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Metric Distributions at Time={time}", fontsize=14, fontweight="bold"
        )
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_temporal_trend(
        metrics_df: pl.DataFrame
        | tuple[pl.DataFrame, str]
        | Sequence[pl.DataFrame | tuple[pl.DataFrame, str]],
        metric: str,
        ax_trend: Optional[matplotlib.axes.Axes] = None,
        ax_violin: Optional[matplotlib.axes.Axes] = None,
        percentile: float = 5.0,
    ) -> tuple[matplotlib.axes.Axes, Optional[matplotlib.axes.Axes]]:
        """Plot temporal trend of a metric across time with mean and quantiles.

        For single dataframe: shows mean line with percentile band across all seeds.
        For multiple experiments: shows mean and quantiles across experiment summaries.
        Optionally shows terminal distribution via violin plot on the right.

        Parameters
        ----------
        metrics_df : pl.DataFrame | tuple[pl.DataFrame, str] | Sequence[pl.DataFrame | tuple[pl.DataFrame, str]]
            Can be any of:
            - Single DataFrame: uses default "Exp 1" label
            - Single (DataFrame, label) tuple: uses provided label
            - Sequence of DataFrames: uses default "Exp 1", "Exp 2", etc. labels
            - Sequence of (DataFrame, label) tuples: uses provided labels
        metric : str
            Column name to plot
        ax_trend : matplotlib.axes.Axes, optional
            Axes for temporal trend plot (creates new if None)
        ax_violin : matplotlib.axes.Axes, optional
            Axes for violin plot (if provided, must share y-axis with ax_trend via GridSpec)
        percentile : float, optional
            Percentile for bands (default: 5.0, shows 5th-95th percentile range)

        Returns
        -------
        tuple[matplotlib.axes.Axes, Optional[matplotlib.axes.Axes]]
            (ax_trend, ax_violin) - both axes or (ax_trend, None) if no violin plot
        """
        # Handle single DataFrame, single tuple, or sequence of DataFrames/tuples
        if isinstance(metrics_df, pl.DataFrame):
            dfs_with_labels = [(metrics_df, None)]
        elif isinstance(metrics_df, tuple):
            # Single (DataFrame, label) tuple
            dfs_with_labels = [metrics_df]
        else:
            # Sequence of DataFrames or (DataFrame, label) tuples
            dfs_with_labels = [
                (item, None) if isinstance(item, pl.DataFrame) else item
                for item in metrics_df
            ]

        if ax_trend is None:
            _, ax_trend = plt.subplots(figsize=(10, 6))

        # Plot each dataframe with its own color
        for idx, (df, label) in enumerate(dfs_with_labels):
            clean_data = df.filter(pl.col(metric).is_finite())
            # Use provided label or default to "Exp N"
            exp_label = label if label is not None else f"Exp {idx + 1}"

            # Compute mean and quantiles across seeds at each time point
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

            # Use color from COLORS list, cycling if necessary
            color = COLORS[idx % len(COLORS)]

            # Plot percentile band
            ax_trend.fill_between(
                times,
                values_lower,
                values_upper,
                alpha=0.2,
                color=color,
                label=f"{exp_label}",
            )

            ax_trend.plot(
                times,
                values_mean,
                color=color,
                linewidth=2.5,
                label=None,  # Skip mean line in legend to reduce clutter
                zorder=10,
            )

            # Plot violin for terminal distribution if requested
            if ax_violin is not None:
                # Get terminal time and values for this experiment
                terminal_time = clean_data[MetricNames.time].max()
                terminal_values = clean_data.filter(
                    pl.col(MetricNames.time) == terminal_time
                )[metric].to_numpy()

                # Clip violin data to percentile range to match temporal trend bounds
                lower_percentile = np.percentile(terminal_values, percentile)
                upper_percentile = np.percentile(terminal_values, 100.0 - percentile)
                terminal_values_clipped = terminal_values[
                    (terminal_values >= lower_percentile)
                    & (terminal_values <= upper_percentile)
                ]

                # Position violins to overlap: center around 1.0
                if len(dfs_with_labels) == 1:
                    pos = 1.0
                else:
                    # Offset based on experiment index, centered around 1.0
                    offset = (idx - (len(dfs_with_labels) - 1) / 2) * 0.15
                    pos = 1.0 + offset

                # Plot violin with overlapping position (no whiskers to avoid redundancy)
                parts = ax_violin.violinplot(
                    [terminal_values_clipped],
                    positions=[pos],
                    widths=0.7,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )

                # Color the violin to match the trend line
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.2)  # Match trend area alpha
                for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
                    if partname in parts:
                        parts[partname].set_color(color)
                        parts[partname].set_linewidth(1.5)

        ax_trend.set_xlabel("Time (hours)", fontsize=11)
        ax_trend.set_ylabel(metric, fontsize=11)
        ax_trend.set_title(
            f"{metric} over Time\n({percentile:.0f}th-{100.0 - percentile:.0f}th percentile)",
            fontsize=12,
            fontweight="bold",
        )
        ax_trend.grid(True, alpha=0.3)
        ax_trend.legend()

        if ax_violin is not None:
            # Since violins overlap, just show a single x-label
            ax_violin.set_xticks([1.0])
            ax_violin.set_xticklabels(["Final\ndistribution"], fontsize=8)
            # Hide y-axis labels on violin plot (trend plot keeps them via sharey)
            ax_violin.tick_params(labelleft=False, left=False)
            ax_violin.grid(True, alpha=0.3, axis="y")

        return (ax_trend, ax_violin)
