from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
from dataclasses import fields, asdict
import itertools
import json
import gc
import os

import numpy as np
import polars as pl

from non_spatial.NonSpatialFusion import _ModelRun
from non_spatial.parametrization import ModelParameters, MetricNames


class MonteCarloEngine:
    """Monte Carlo simulation engine for non-spatial fusion model."""

    @staticmethod
    def monte_carlo_simulation(
        parameters: ModelParameters,
        seeds: list[int],
        save_path: Path,
        batch_size: int = 10000,
    ):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        temp_dir = save_path / ".temp_batches"
        temp_dir.mkdir(exist_ok=True)

        num_batches = (len(seeds) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(seeds))
            batch_seeds = seeds[start_idx:end_idx]

            lineage_data_all, metrics_data_all, all_genotypes_all = (
                _run_monte_carlo_simulation(
                    Ngenes=parameters.number_of_genes,
                    NgenerationsMax=parameters.number_of_generations,
                    DT=parameters.dt,
                    DATA_RESOLUTION=parameters.data_resolution,
                    KC=parameters.carrying_capacity,
                    pm=parameters.mutation_rate_per_gene,
                    pf=parameters.fusion_rate,
                    growthRate=parameters.growth_rate,
                    deathRate=parameters.death_rate,
                    diversity=parameters.diversity
                    if parameters.diversity is not None
                    else 0,
                    initial_population_size=parameters.initial_population_size,
                    treatment_injection_every=parameters.treatment_injection_every
                    if parameters.treatment_injection_every is not None
                    else -1,
                    treatment_initial_concentration=parameters.treatment_initial_concentration,
                    treatment_halflife=parameters.treatment_halflife,
                    treatment_concentration_to_extra_death=(
                        parameters.treatment_concentration_to_extra_death
                    ),
                    treatment_selection=parameters.treatment_selection,
                    treatment_resistivity=parameters.treatment_resistivity,
                    seeds=batch_seeds,
                )
            )

            lineage_seed_ids = []
            lineage_times = []
            lineage_cell_counts = []
            lineage_genotype_ids = []
            lineage_ancestor_ids = []

            metrics_seed_ids = []
            metrics_times = []
            metrics_total_cells = []
            metrics_num_genotypes = []
            metrics_shannon = []
            metrics_simpson = []
            metrics_max_mutations = []
            metrics_drug_concentration = []
            metrics_drug_extra_death_wt = []

            genotype_seed_ids = []
            genotype_ids = []
            genotype_mutations = []
            genotype_arrays = []

            for i, seed in enumerate(batch_seeds):
                if lineage_data_all[i]:
                    lineage_array = np.array(lineage_data_all[i])
                    lineage_seed_ids.extend([seed] * len(lineage_array))
                    lineage_times.extend(lineage_array[:, 0].astype(float))
                    lineage_cell_counts.extend(lineage_array[:, 1].astype(int))
                    lineage_genotype_ids.extend(lineage_array[:, 2].astype(int))
                    lineage_ancestor_ids.extend(lineage_array[:, 3].astype(int))

                if metrics_data_all[i]:
                    metrics_array = np.array(metrics_data_all[i])
                    metrics_seed_ids.extend([seed] * len(metrics_array))
                    metrics_times.extend(metrics_array[:, 0].astype(float))
                    metrics_total_cells.extend(metrics_array[:, 1].astype(int))
                    metrics_num_genotypes.extend(metrics_array[:, 2].astype(int))
                    metrics_shannon.extend(metrics_array[:, 3].astype(float))
                    metrics_simpson.extend(metrics_array[:, 4].astype(float))
                    metrics_max_mutations.extend(metrics_array[:, 5].astype(int))
                    metrics_drug_concentration.extend(metrics_array[:, 6].astype(float))
                    metrics_drug_extra_death_wt.extend(metrics_array[:, 7].astype(float))

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

            if lineage_seed_ids:
                lineage_df = pl.DataFrame(
                    {
                        MetricNames.seed: lineage_seed_ids,
                        MetricNames.time: lineage_times,
                        MetricNames.cell_count: lineage_cell_counts,
                        MetricNames.genotype_id: lineage_genotype_ids,
                        MetricNames.ancestor_id: lineage_ancestor_ids,
                    }
                )
                lineage_df.write_parquet(
                    temp_dir / f"lineage_batch_{batch_idx}.parquet"
                )
                del (
                    lineage_df,
                    lineage_seed_ids,
                    lineage_times,
                    lineage_cell_counts,
                    lineage_genotype_ids,
                    lineage_ancestor_ids,
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
                        MetricNames.drug_concentration: metrics_drug_concentration,
                        MetricNames.drug_extra_death_wt: metrics_drug_extra_death_wt,
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
                    metrics_drug_concentration,
                    metrics_drug_extra_death_wt,
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

            del lineage_data_all, metrics_data_all, all_genotypes_all
            gc.collect()

        _concatenate_batch_files(temp_dir, save_path, num_batches)
        shutil.rmtree(temp_dir)

    @staticmethod
    def monte_carlo_parameter_sweep(
        parameters: ModelParameters,
        seeds: list[int],
        sweep_params: dict[str, np.ndarray],
        save_path: Path,
        batch_size: int = 10000,
    ):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        field_names = {f.name for f in fields(ModelParameters)}
        invalid_keys = set(sweep_params.keys()) - field_names
        if invalid_keys:
            raise ValueError(
                f"Invalid sweep parameters: {invalid_keys}. "
                f"Valid options: {sorted(field_names)}"
            )

        param_names = sorted(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]
        combinations = list(itertools.product(*param_values))

        sweep_metadata = {
            "param_names": param_names,
            "param_grids": {name: sweep_params[name].tolist() for name in param_names},
            "num_combinations": len(combinations),
            "num_seeds": len(seeds),
            "total_runs": len(combinations) * len(seeds),
        }

        results_dirs = []
        for combo_values in combinations:
            combo_params = _update_model_parameters(
                parameters, param_names, combo_values
            )

            combo_dir_name = "_".join(
                f"{name}={val:.4g}" for name, val in zip(param_names, combo_values)
            )
            combo_dir = save_path / combo_dir_name
            combo_dir.mkdir(parents=True, exist_ok=True)

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

            del combo_params, combo_dir
            gc.collect()

        sweep_metadata["results"] = results_dirs
        metadata_path = save_path / "sweep_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sweep_metadata, f, indent=2)


def _update_model_parameters(
    base_params: ModelParameters,
    param_names: list[str],
    param_values: tuple,
) -> ModelParameters:
    params_dict = asdict(base_params)
    for name, value in zip(param_names, param_values):
        params_dict[name] = value
    return ModelParameters(**params_dict)


def _concatenate_batch_files(temp_dir: Path, save_path: Path, num_batches: int) -> None:
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
    DT: float,
    DATA_RESOLUTION: int,
    KC: int,
    pm: float,
    pf: float,
    growthRate: float,
    deathRate: float,
    diversity: int,
    initial_population_size: int,
    treatment_injection_every: int,
    treatment_initial_concentration: float,
    treatment_halflife: float,
    treatment_concentration_to_extra_death: float,
    treatment_selection: float,
    treatment_resistivity: float,
    seeds: list[int],
) -> tuple[list, list, list]:
    def run_single_seed(seed: int):
        np.random.seed(int(seed))
        return _ModelRun(
            Ngenes=Ngenes,
            NgenerationsMax=NgenerationsMax,
            DT=DT,
            DATA_RESOLUTION=DATA_RESOLUTION,
            KC=KC,
            pm=pm,
            pf=pf,
            growthRate=growthRate,
            deathRate=deathRate,
            diversity=diversity,
            initial_population_size=initial_population_size,
            treatment_injection_every=treatment_injection_every,
            treatment_initial_concentration=treatment_initial_concentration,
            treatment_halflife=treatment_halflife,
            treatment_concentration_to_extra_death=treatment_concentration_to_extra_death,
            treatment_selection=treatment_selection,
            treatment_resistivity=treatment_resistivity,
        )

    max_workers = min(len(seeds), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_seed, seeds))

    lineage_data_all = [r[0] for r in results]
    metrics_data_all = [r[1] for r in results]
    all_genotypes_all = [r[2] for r in results]

    return lineage_data_all, metrics_data_all, all_genotypes_all