from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
from typing import get_type_hints
from dataclasses import fields, asdict
import itertools
import json
import gc

import numpy as np
import polars as pl
from numba import njit

from non_spatial.NonSpatialFusion import _ModelRun
from non_spatial.parametrization import ModelParameters, MetricNames


class MonteCarloEngine:
    @staticmethod
    def monte_carlo_simulation(
        parameters: ModelParameters,
        seeds: list[int],
        save_path: Path,
        batch_size: int = 10000,
    ):
        """Run Monte Carlo simulations with memory-efficient batch processing.

        Parameters
        ----------
        parameters : ModelParameters
            Model parameters
        seeds : list[int]
            List of random seeds to run
        save_path : Path
            Directory to save results
        batch_size : int, optional
            Number of seeds to process in parallel per batch (default: 10000)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create temp directory for batch files
        temp_dir = save_path / ".temp_batches"
        temp_dir.mkdir(exist_ok=True)

        # Process seeds in batches
        num_batches = (len(seeds) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(seeds))
            batch_seeds = seeds[start_idx:end_idx]

            # Run batch in parallel
            lineage_data_all, metrics_data_all, all_genotypes_all = (
                _run_monte_carlo_simulation(
                    Ngenes=parameters.number_of_genes,
                    NgenerationsMax=parameters.number_of_generations,
                    DT=parameters.dt,
                    KC=parameters.carrying_capacity,
                    pm=parameters.mutation_rate_per_gene,
                    pf=parameters.fusion_rate,
                    growthRate=parameters.growth_rate,
                    deathRate=parameters.death_rate,
                    diversity=parameters.diversity,
                    treatment_every=parameters.treatment_every
                    if parameters.treatment_every is not None
                    else -1,
                    treatment_duration=parameters.treatment_duration,
                    treatment_base_extra=parameters.treatment_base_extra_death,
                    treatment_selection=parameters.treatment_selection,
                    treatment_cell_density_dependence=parameters.treatment_cell_density_dependence,
                    seeds=batch_seeds,
                )
            )

            # Flatten results from this batch (vectorized for speed)
            lineage_seed_ids = []
            lineage_times = []
            lineage_births = []
            lineage_deaths = []
            lineage_fusions = []

            metrics_seed_ids = []
            metrics_times = []
            metrics_total_cells = []
            metrics_num_genotypes = []
            metrics_shannon = []
            metrics_simpson = []
            metrics_max_mutations = []

            genotype_seed_ids = []
            genotype_ids = []
            genotype_mutations = []
            genotype_arrays = []

            for i, seed in enumerate(batch_seeds):
                # Convert Numba lists to numpy arrays for vectorized access
                if lineage_data_all[i]:
                    lineage_array = np.array(lineage_data_all[i])
                    lineage_seed_ids.extend([seed] * len(lineage_array))
                    lineage_times.extend(lineage_array[:, 0].astype(int))
                    lineage_births.extend(lineage_array[:, 1].astype(int))
                    lineage_deaths.extend(lineage_array[:, 2].astype(int))
                    lineage_fusions.extend(lineage_array[:, 3].astype(int))

                if metrics_data_all[i]:
                    metrics_array = np.array(metrics_data_all[i])
                    metrics_seed_ids.extend([seed] * len(metrics_array))
                    metrics_times.extend(metrics_array[:, 0].astype(int))
                    metrics_total_cells.extend(metrics_array[:, 1].astype(int))
                    metrics_num_genotypes.extend(metrics_array[:, 2].astype(int))
                    metrics_shannon.extend(metrics_array[:, 3].astype(float))
                    metrics_simpson.extend(metrics_array[:, 4].astype(float))
                    metrics_max_mutations.extend(metrics_array[:, 5].astype(int))

                if all_genotypes_all[i]:
                    genotypes_list = all_genotypes_all[i]
                    genotype_mutations_array = np.array(
                        [np.sum(g) for g in genotypes_list], dtype=int
                    )
                    genotype_arrays_hex = [g.tobytes().hex() for g in genotypes_list]

                    genotype_seed_ids.extend([seed] * len(genotypes_list))
                    genotype_ids.extend(range(len(genotypes_list)))
                    genotype_mutations.extend(genotype_mutations_array)
                    genotype_arrays.extend(genotype_arrays_hex)

            # Write batch files
            if lineage_seed_ids:
                lineage_df = pl.DataFrame(
                    {
                        MetricNames.seed: lineage_seed_ids,
                        MetricNames.time: lineage_times,
                        MetricNames.births: lineage_births,
                        MetricNames.deaths: lineage_deaths,
                        MetricNames.fusions: lineage_fusions,
                    }
                )
                lineage_df.write_parquet(
                    temp_dir / f"lineage_batch_{batch_idx}.parquet"
                )
                del (
                    lineage_df,
                    lineage_seed_ids,
                    lineage_times,
                    lineage_births,
                    lineage_deaths,
                    lineage_fusions,
                )

            if metrics_seed_ids:
                metrics_df = pl.DataFrame(
                    {
                        MetricNames.seed: metrics_seed_ids,
                        MetricNames.time: metrics_times,
                        MetricNames.total_cells: metrics_total_cells,
                        MetricNames.num_genotypes: metrics_num_genotypes,
                        MetricNames.shannon_index: metrics_shannon,
                        MetricNames.simpson_index: metrics_simpson,
                        MetricNames.max_mutations: metrics_max_mutations,
                    }
                )
                metrics_df.write_parquet(
                    temp_dir / f"metrics_batch_{batch_idx}.parquet"
                )
                del (
                    metrics_df,
                    metrics_seed_ids,
                    metrics_times,
                    metrics_total_cells,
                    metrics_num_genotypes,
                    metrics_shannon,
                    metrics_simpson,
                    metrics_max_mutations,
                )

            if genotype_seed_ids:
                genotypes_df = pl.DataFrame(
                    {
                        MetricNames.seed: genotype_seed_ids,
                        MetricNames.genotype_id: genotype_ids,
                        MetricNames.mutations: genotype_mutations,
                        MetricNames.genotype_array: genotype_arrays,
                    }
                )
                genotypes_df.write_parquet(
                    temp_dir / f"genotypes_batch_{batch_idx}.parquet"
                )
                del (
                    genotypes_df,
                    genotype_seed_ids,
                    genotype_ids,
                    genotype_mutations,
                    genotype_arrays,
                )

            # Clean up simulation results and trigger garbage collection
            del lineage_data_all, metrics_data_all, all_genotypes_all
            gc.collect()

        # Concatenate all batch files into final outputs
        _concatenate_batch_files(temp_dir, save_path, num_batches)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

    @staticmethod
    def monte_carlo_parameter_sweep(
        parameters: ModelParameters,
        seeds: list[int],
        sweep_params: dict[str, np.ndarray],
        save_path: Path,
        batch_size: int = 10000,
    ):
        """Run Monte Carlo simulations across a parameter sweep grid.

        Parameters
        ----------
        parameters : ModelParameters
            Template parameters (all required fields must be set)
        seeds : list[int]
            List of random seeds (same for all parameter combinations)
        sweep_params : dict[str, np.ndarray]
            Parameter sweep grid. Keys must match ModelParameters field names,
            values are 1D arrays of values to sweep through.
        save_path : Path
            Directory to save results
        batch_size : int, optional
            Number of seeds per batch (default: 10000)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Validate sweep_params keys against ModelParameters
        type_hints = get_type_hints(ModelParameters)
        field_names = {f.name for f in fields(ModelParameters)}
        invalid_keys = set(sweep_params.keys()) - field_names
        if invalid_keys:
            raise ValueError(
                f"Invalid sweep parameters: {invalid_keys}. "
                f"Valid options: {sorted(field_names)}"
            )

        # Generate all parameter combinations
        param_names = sorted(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]
        combinations = list(itertools.product(*param_values))

        # Store metadata about the sweep
        sweep_metadata = {
            "param_names": param_names,
            "param_grids": {name: sweep_params[name].tolist() for name in param_names},
            "num_combinations": len(combinations),
            "num_seeds": len(seeds),
            "total_runs": len(combinations) * len(seeds),
        }

        # Run each parameter combination
        results_dirs = []
        for combo_idx, combo_values in enumerate(combinations):
            # Create a copy of base parameters
            combo_params = _update_model_parameters(
                parameters, param_names, combo_values
            )

            # Create output directory for this combination
            combo_dir_name = "_".join(
                f"{name}={val:.4g}" for name, val in zip(param_names, combo_values)
            )
            combo_dir = save_path / combo_dir_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Run simulation for this combination
            MonteCarloEngine.monte_carlo_simulation(
                parameters=combo_params,
                seeds=seeds,
                save_path=combo_dir,
                batch_size=batch_size,
            )

            results_dirs.append(
                {
                    "combination": {
                        name: float(val) for name, val in zip(param_names, combo_values)
                    },
                    "output_dir": str(combo_dir),
                }
            )

            # Clean up between parameter combinations
            del combo_params, combo_dir
            gc.collect()

        # Save sweep metadata
        sweep_metadata["results"] = results_dirs
        metadata_path = save_path / "sweep_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sweep_metadata, f, indent=2)


def _update_model_parameters(
    base_params: ModelParameters,
    param_names: list[str],
    param_values: tuple,
) -> ModelParameters:
    """Create a new ModelParameters with specified fields updated.

    Parameters
    ----------
    base_params : ModelParameters
        Template parameters to update
    param_names : list[str]
        Names of fields to update
    param_values : tuple
        Values corresponding to param_names

    Returns
    -------
    ModelParameters
        New ModelParameters instance with updates applied
    """
    params_dict = asdict(base_params)
    for name, value in zip(param_names, param_values):
        params_dict[name] = value
    return ModelParameters(**params_dict)


def _concatenate_batch_files(temp_dir: Path, save_path: Path, num_batches: int) -> None:
    """Concatenate all batch parquet files into final output files."""
    # Concatenate lineage data
    lineage_batches = []
    for i in range(num_batches):
        batch_file = temp_dir / f"lineage_batch_{i}.parquet"
        if batch_file.exists():
            lineage_batches.append(pl.read_parquet(batch_file))
    if lineage_batches:
        lineage_df = pl.concat(lineage_batches)
        lineage_df.write_parquet(save_path / "lineage_data.parquet")
        del lineage_df, lineage_batches
        gc.collect()

    # Concatenate metrics data
    metrics_batches = []
    for i in range(num_batches):
        batch_file = temp_dir / f"metrics_batch_{i}.parquet"
        if batch_file.exists():
            metrics_batches.append(pl.read_parquet(batch_file))
    if metrics_batches:
        metrics_df = pl.concat(metrics_batches)
        metrics_df.write_parquet(save_path / "metrics_data.parquet")
        del metrics_df, metrics_batches
        gc.collect()

    # Concatenate genotypes data
    genotypes_batches = []
    for i in range(num_batches):
        batch_file = temp_dir / f"genotypes_batch_{i}.parquet"
        if batch_file.exists():
            genotypes_batches.append(pl.read_parquet(batch_file))
    if genotypes_batches:
        genotypes_df = pl.concat(genotypes_batches)
        genotypes_df.write_parquet(save_path / "genotypes_data.parquet")
        del genotypes_df, genotypes_batches
        gc.collect()


def _run_monte_carlo_simulation(
    Ngenes: int,
    NgenerationsMax: int,
    DT: int,
    KC: int,
    pm: float,
    pf: float,
    growthRate: float,
    deathRate: float,
    diversity: int,
    treatment_every: int,
    treatment_duration: int,
    treatment_base_extra: float,
    treatment_selection: float,
    treatment_cell_density_dependence: float,
    seeds: list[int],
) -> tuple[list, list, list]:
    def run_single_seed(seed: int):
        """Run a single model with the given seed."""
        np.random.seed(int(seed))
        return _ModelRun(
            Ngenes=Ngenes,
            NgenerationsMax=NgenerationsMax,
            DT=DT,
            KC=KC,
            pm=pm,
            pf=pf,
            growthRate=growthRate,
            deathRate=deathRate,
            diversity=diversity,
            treatment_every=treatment_every,
            treatment_duration=treatment_duration,
            treatment_base_extra=treatment_base_extra,
            treatment_selection=treatment_selection,
            treatment_cell_density_dependence=treatment_cell_density_dependence,
        )

    # Run simulations in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        results = list(executor.map(run_single_seed, seeds))

    # Unpack results
    lineage_data_all = [r[0] for r in results]
    metrics_data_all = [r[1] for r in results]
    all_genotypes_all = [r[2] for r in results]

    return lineage_data_all, metrics_data_all, all_genotypes_all
