"""'marimo edit non_spatial/monte_carlo/test_monte_carlo.py'"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import tempfile
    import marimo

    # Add project root to path for imports
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.monte_carlo.monte_carlo import MonteCarloEngine
    from non_spatial.parametrization import ModelParameters

    return ModelParameters, MonteCarloEngine, Path, marimo, tempfile


@app.cell
def _(ModelParameters, Path, marimo, tempfile):
    # Create a temporary output directory
    output_dir = Path(tempfile.mkdtemp(prefix="mc_test_"))

    # Set up test parameters with small values for quick testing
    params = ModelParameters(
        number_of_genes=5,
        carrying_capacity=100,
        number_of_generations=50,
        mutation_rate_per_gene=0.01,
        fusion_rate=0.001,
        growth_rate=0.5,
        death_rate=0.2,
        save_path=output_dir,
        dt=1,
        diversity=3,
    )

    marimo.md(
        f"**Test Parameters:**\n- Output: {output_dir}\n- Genes: {params.number_of_genes}\n- Generations: {params.number_of_generations}"
    )
    return output_dir, params


@app.cell
def _(MonteCarloEngine, marimo, output_dir, params):
    # Create 3 seeds for quick test
    seeds = [i for i in range(0, 1000)]

    marimo.md(f"**Running Monte Carlo simulation with {len(seeds)} seeds...**")

    # Run the simulation
    MonteCarloEngine.monte_carlo_simulation(
        parameters=params,
        seeds=seeds,
        save_path=output_dir,
    )

    marimo.md("✅ Simulation completed!")
    return


@app.cell
def _(marimo, output_dir):
    import polars as pl

    # Check output files
    output_files = list(output_dir.glob("*.parquet"))

    marimo.md(f"**Output files created:** {len(output_files)}")

    results = {}
    for file in output_files:
        df = pl.read_parquet(file)
        results[file.name] = df
        marimo.md(f"\n**{file.name}** ({df.shape[0]} rows)")
        marimo.md(f"Columns: {', '.join(df.columns)}")

    results
    return (results,)


@app.cell
def _(marimo, results):
    # Show sample data from each file
    if "lineage_data.parquet" in results:
        marimo.md("**Lineage Data Sample:**")
        results["lineage_data.parquet"].head(3)
    else:
        marimo.md("No lineage data")
    return


@app.cell
def _(marimo, results):
    if "metrics_data.parquet" in results:
        marimo.md("**Metrics Data Sample:**")
        results["metrics_data.parquet"].head(3)
    else:
        marimo.md("No metrics data")
    return


@app.cell
def _(marimo, results):
    if "genotypes_data.parquet" in results:
        marimo.md("**Genotypes Data Sample:**")
        results["genotypes_data.parquet"].head(3)
    else:
        marimo.md("No genotypes data")
    return


if __name__ == "__main__":
    app.run()
