"""marimo edit tests/test_monte_carlo_param_sweep.py"""

import marimo

__generated_with = "0.20.4"
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
    from non_spatial.parametrization import (
        ModelParameters,
        MetricNames,
        ModelParametersTyping,
    )
    from tests.test_output import TEST_OUTPUT_PATH

    return (
        ModelParameters,
        ModelParametersTyping,
        MonteCarloEngine,
        TEST_OUTPUT_PATH,
        marimo,
    )


@app.cell
def _(ModelParameters, ModelParametersTyping, TEST_OUTPUT_PATH, marimo):
    import numpy as np

    # Define parameter sweep configuration
    output_dir = TEST_OUTPUT_PATH / "param_sweep_4"

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

    sweep_params: dict[ModelParametersTyping, np.ndarray] = {
        "mutation_rate_per_gene": np.array([1e-6, 1e-5, 1e-4]),
    }

    from typing import get_args

    fixed_parameters = "\n\n".join(
        [
            f"{i}={getattr(base_params, i)}"
            for i in get_args(ModelParametersTyping)
            if i not in sweep_params
        ]
    )

    marimo.md(f"**Fixed Parameters**:\n\n{fixed_parameters}")
    return base_params, output_dir, sweep_params


@app.cell
def _(
    MonteCarloEngine,
    base_params,
    marimo,
    output_dir,
    sweep_params: "dict[ModelParametersTyping, np.ndarray]",
):
    seeds = list(range(300))

    try:
        MonteCarloEngine.monte_carlo_parameter_sweep(
            parameters=base_params,
            seeds=seeds,
            sweep_params=sweep_params,
            save_path=output_dir,
        )
        _output = "Parameter sweep completed!"
    except Exception as e:
        _output = f"Sweep failed:\n\n{e}"
    marimo.md(_output)
    return


if __name__ == "__main__":
    app.run()
