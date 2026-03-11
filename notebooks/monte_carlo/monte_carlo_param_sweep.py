"""marimo edit notebooks/monte_carlo/monte_carlo_param_sweep.py

Note: this notebook only contains the code that performs the parameter sweep. To visualize the results,
use the notebook `notebooks/monte_carlo_param_sweep_visualization.py` which is designed to read the output of this sweep and generate plots.

To perform Monte Carlo simulation for single parameter set, we just sweep over one parameter with one value.

Usage: set the 'output_dir', parameters, seeds to run and sweep to perform. The sweep to perform
is determined by the 'sweep_params' dictionary, with keys of type 'ModelParametersTyping' and values of type 'List[float]'.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo

    # Add project root to path for imports
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))

    from non_spatial.monte_carlo.monte_carlo import MonteCarloEngine
    from non_spatial.parametrization import (
        ModelParameters,
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
        number_of_genes=100,
        carrying_capacity=3000,
        number_of_generations=24 * 4 * 140,  # 140 days, 15-minute steps
        mutation_rate_per_gene=1e-4,  # per birth event -> unchanged
        fusion_rate=1e-4 / 48.0,  # rescaled from per-12h to per-15min
        growth_rate=0.12 / 48.0,  # rescaled from per-12h to per-15min
        death_rate=0.04 / 48.0,  # rescaled from per-12h to per-15min
        save_path=output_dir,
        dt=0.25,  # 15 minutes = 0.25 hours
        data_resolution=24 * 4,  # store every day to prevent file size from exploding
        diversity=1,
        seed=0,
        treatment_injection_every=21 * 24 * 4,  # every 3 weeks
        treatment_initial_concentration=0.25,
        treatment_halflife=12.0,  # 12 hours
        treatment_concentration_to_extra_death=0.7 / 48.0,  # per 15 min
        treatment_selection=0.1,
        treatment_resistivity=1.0,
    )
    sweep_params: dict[ModelParametersTyping, np.ndarray] = {
        "mutation_rate_per_gene": np.array([1e-4]),
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
