"""Testing the plotting of results"""

import pytest
import polars as pl

from non_spatial.parametrization import MetricNames
from non_spatial.plot_results import (
    plot_metric,
    plot_lineage_stack,
    plot_genotype_heatmap,
)
from tests.test_files import TEST_FILES
from tests.test_output import TEST_OUTPUT_PATH


@pytest.fixture
def fixture_metrics_data() -> pl.DataFrame:
    """Fixture for testing the plotting of results."""
    return pl.read_csv(source=TEST_FILES / "MetricsNone.csv")


@pytest.fixture
def fixture_lineage_data() -> pl.DataFrame:
    """Fixture for testing the plotting of lineage data results"""
    return pl.read_csv(source=TEST_FILES / "LineageNone.csv")


@pytest.fixture
def fixture_genotype_data() -> pl.DataFrame:
    """Fixture for testing the plotting of genotype data results"""
    return pl.read_csv(source=TEST_FILES / "GenotypesNone.csv")


class TestPlotMetric:
    """Testing the plot_metric function"""

    def test_plot_metric_cell_count_data(
        self, fixture_metrics_data: pl.DataFrame
    ) -> None:
        """Test the plotting of results."""
        cell_count_data = fixture_metrics_data.select(
            [MetricNames.time, MetricNames.cell_count]
        )
        plot_metric(
            data=cell_count_data,
            plot_title="Total Cells Over Time",
            x_label="Time",
            y_label="Total Cells",
            save_path=TEST_OUTPUT_PATH / "plot_total_cells",
        )

    def test_plot_metric_num_genotypes(
        self, fixture_metrics_data: pl.DataFrame
    ) -> None:
        """Test the plotting of results with number of genotypes."""
        num_genotypes_data = fixture_metrics_data.select(
            [MetricNames.time, MetricNames.num_genotypes]
        )
        plot_metric(
            data=num_genotypes_data,
            plot_title="Number of Genotypes Over Time",
            x_label="Time",
            y_label="Number of Genotypes",
            save_path=TEST_OUTPUT_PATH / "plot_num_genotypes",
        )

    def test_plot_metric_simpson(self, fixture_metrics_data: pl.DataFrame) -> None:
        """Test the plotting of Simpson."""
        diversity_data = fixture_metrics_data.select(
            [MetricNames.time, MetricNames.simpson_index]
        )
        plot_metric(
            data=diversity_data,
            plot_title="Simpson Index Over Time",
            x_label="Time",
            y_label="Simpson Index",
            save_path=TEST_OUTPUT_PATH / "plot_simpson",
        )

    def test_plot_metric_shannon(self, fixture_metrics_data: pl.DataFrame) -> None:
        """Test the plotting of Shannon."""
        diversity_data = fixture_metrics_data.select(
            [MetricNames.time, MetricNames.shannon_index]
        )
        plot_metric(
            data=diversity_data,
            plot_title="Shannon Index Over Time",
            x_label="Time",
            y_label="Shannon Index",
            save_path=TEST_OUTPUT_PATH / "plot_shannon",
        )

    def test_plot_metric_max_mutations(
        self, fixture_metrics_data: pl.DataFrame
    ) -> None:
        """Test the plotting of results with max mutations."""
        max_mutations_data = fixture_metrics_data.select(
            [MetricNames.time, MetricNames.max_mutations]
        )
        plot_metric(
            data=max_mutations_data,
            plot_title="Max Mutations Over Time",
            x_label="Time",
            y_label="Max Mutations",
            save_path=TEST_OUTPUT_PATH / "plot_max_mutations",
        )


class TestPlotLineageStack:
    """Testing the plot_lineage_stack function"""

    def test_plot_lineage_stacked_area(
        self, fixture_lineage_data: pl.DataFrame
    ) -> None:
        """Test the plotting of lineage composition as a stacked area plot."""
        plot_lineage_stack(
            data=fixture_lineage_data,
            save_path=TEST_OUTPUT_PATH / "plot_lineage_composition",
            top_n=10,
        )


class TestGenotypeHeatmap:
    """Testing the plot_genotype_heatmap function"""

    def test_plot_genotype_heatmap(self, fixture_genotype_data: pl.DataFrame) -> None:
        """Test the plotting of genotype heatmap."""
        plot_genotype_heatmap(
            data=fixture_genotype_data,
            save_path=TEST_OUTPUT_PATH / "plot_genotype_heatmap",
        )
