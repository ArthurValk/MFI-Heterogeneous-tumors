"""Module for plotting results from In silico experiments"""

from pathlib import Path
from typing import Literal
import matplotlib as plt
import numpy as np
import polars as pl
from matplotlib import use

from non_spatial_no_chemotherapy.parametrization import MetricNames

use("Agg")  # Use non-interactive backend for plotting in tests


def plot_metric(
    data: pl.DataFrame,
    plot_title: str,
    x_label: str,
    y_label: str,
    save_path: Path,
    extension: Literal["png", "pdf"] = "png",
) -> None:
    """Plot metrics over time

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the metrics to plot. The first column should be the x-axis (e.g., time),
        and the remaining columns should be the metrics to plot on the y-axis.
    plot_title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    save_path : Path
        Path to save the plot (without extension).
    extension : Literal["png", "pdf"], optional
        File extension for the saved plot. Can be either "png" or "pdf".
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get column names
    columns = data.columns
    x_col = columns[0]
    y_cols = columns[1:]

    # Plot each metric column
    x_data = data[x_col].to_numpy()
    for y_col in y_cols:
        y_data = data[y_col].to_numpy()
        ax.plot(x_data, y_data, label=y_col, marker="o", markersize=5, linewidth=2)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")

    if len(y_cols) > 0:
        ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3, linestyle="--")

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with proper extension
    file_path = save_path.with_suffix(f".{extension}")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lineage_stack(
    data: pl.DataFrame,
    save_path: Path,
    extension: Literal["png", "pdf"] = "png",
    top_n: int = 20,
) -> None:
    """Plotting lineage data as a stacked area plot.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing lineage data with Time, GenotypeId, and CellCount columns.
    save_path : Path
        Path to save the plot (without extension).
    extension : Literal["png", "pdf"], optional
        File extension for the saved plot. Can be either "png" or "pdf".
    top_n : int, optional
        Only plot the top N genotypes by total cell count. Default is 20.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by Time and GenotypeId, sum CellCount
    grouped = (
        data.group_by([MetricNames.time, MetricNames.genotype_id])
        .agg(pl.col(MetricNames.cell_count).sum())
        .sort([MetricNames.time, MetricNames.genotype_id])
    )

    # Filter to top N genotypes at each time point
    # This captures dynamic diversity without missing important genotypes at specific times
    grouped = (
        grouped.with_columns(
            pl.col(MetricNames.cell_count)
            .rank(descending=True)
            .over(MetricNames.time)
            .alias("rank_at_time")
        )
        .filter(pl.col("rank_at_time") <= top_n)
        .drop("rank_at_time")
    )

    # Convert grouped data to dict for fast lookup
    grouped_dict = grouped.to_dict(as_series=False)
    times_list = grouped_dict[MetricNames.time]
    genotypes_list = grouped_dict[MetricNames.genotype_id]
    counts_list = grouped_dict[MetricNames.cell_count]

    # Build lookup: (Time, GenotypeId) -> CellCount
    lookup = dict(zip(zip(times_list, genotypes_list), counts_list))

    # Get unique times and genotypes
    times = sorted(set(times_list))
    genotypes = sorted(set(genotypes_list))

    # Build data for stackplot: for each genotype, get CellCount at each time
    stackplot_data = []
    labels = []
    for genotype in genotypes:
        counts = [lookup.get((t, genotype), 0) for t in times]
        stackplot_data.append(counts)
        labels.append(f"Genotype {genotype}")

    # Create stacked area plot
    ax.stackplot(times, *stackplot_data, labels=labels, alpha=0.8)

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Cell Count", fontsize=12)
    ax.set_title("Lineage Composition Over Time", fontsize=14, fontweight="bold")
    # ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with proper extension
    file_path = save_path.with_suffix(f".{extension}")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_genotype_heatmap(
    data: pl.DataFrame,
    save_path: Path,
    extension: Literal["png", "pdf"] = "png",
    genotype_filter: list[int] | int = 200,
) -> None:
    """Visualize genotype data as a binary heatmap.

    Each row represents a genotype, each column represents a locus.
    Purple indicates wild-type (False), yellow indicates mutated (True).

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing genotype data with GenotypeId and Locus0, Locus1, ... columns.
        Locus columns should contain boolean values (True/False).
    save_path : Path
        Path to save the plot (without extension).
    extension : Literal["png", "pdf"], optional
        File extension for the saved plot. Can be either "png" or "pdf".
    genotype_filter : list[int] | int, optional
        Specifies which genotypes to plot:
        - None: Plot all genotypes
        - int: Plot only genotypes with ID < genotype_filter (first N genotypes)
        - list[int]: Plot only the specified genotype IDs
    """
    # Filter genotypes based on genotype_filter parameter
    if genotype_filter is not None:
        if isinstance(genotype_filter, int):
            # Filter to first N genotypes by ID
            data = data.filter(pl.col("GenotypeId") < genotype_filter)
        elif isinstance(genotype_filter, list):
            # Filter to specific genotype IDs
            data = data.filter(pl.col("GenotypeId").is_in(genotype_filter))

    # Extract locus columns (all columns except GenotypeId)
    locus_columns = [col for col in data.columns if col.startswith("Locus")]
    genotype_ids = data["GenotypeId"].to_list()

    # Convert boolean values to numeric (True=1, False=0)
    locus_data = data.select(locus_columns).cast(pl.Int32).to_numpy()

    # Create figure and axes
    num_genotypes = len(genotype_ids)
    num_loci = len(locus_columns)
    figsize = (max(12, num_loci / 3), max(6, num_genotypes / 2))
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    ax.imshow(
        locus_data,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )

    # Add gridlines to delineate rows and columns
    ax.set_xticks(np.arange(-0.5, num_loci, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_genotypes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(which="minor", size=0)

    # Configure x-axis (loci)
    ax.set_xlabel("Locus", fontsize=12)
    locus_indices = np.arange(0, num_loci, max(1, num_loci // 10))
    ax.set_xticks(locus_indices)
    ax.set_xticklabels([int(i) for i in locus_indices], fontsize=9)

    # Configure y-axis (genotypes) with smart labeling
    ax.set_ylabel("Genotype ID", fontsize=12)
    if num_genotypes <= 50:
        # Show all genotype labels if reasonable number
        ax.set_yticks(np.arange(num_genotypes))
        ax.set_yticklabels(genotype_ids, fontsize=8)
    else:
        # Show subset of labels for many genotypes
        label_interval = max(1, num_genotypes // 20)
        label_indices = np.arange(0, num_genotypes, label_interval)
        ax.set_yticks(label_indices)
        ax.set_yticklabels([genotype_ids[int(i)] for i in label_indices], fontsize=8)

    ax.set_title("Genotype Mutation Patterns", fontsize=14, fontweight="bold")

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with proper extension
    file_path = save_path.with_suffix(f".{extension}")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
