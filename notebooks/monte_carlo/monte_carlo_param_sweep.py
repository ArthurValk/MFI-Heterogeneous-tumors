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
        carrying_capacity=1000,
        dt=1,  # 1 hour per generation
        number_of_generations=4320,  # 180 days * 24 hours
        mutation_rate_per_gene=5.2e-5,  # Rescaled from 1e-4 (factor: 1/1.92)
        fusion_rate=5.2e-5,  # Rescaled from 1e-4 (factor: 1/1.92)
        growth_rate=0.02,  # birth = 1.5 * death, birth - death = 0.08/12
        death_rate=0.08 / 6,  # death = 0.08/12 / 0.5 = 0.08/6
        save_path=output_dir,
        treatment_every=156,  # 20*24 - 2 (hours between treatments)
        treatment_duration=12,  # 2 hours
        treatment_base_extra_death=2
        * 0.16
        / 12,  # mean 0.16 per 12 hours, therefore max=2*0.16
        treatment_selection=0.1,
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
