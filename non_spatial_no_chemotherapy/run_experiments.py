"""File to coordinate the running of experiments for the non-spatial model."""

from pathlib import Path

from non_spatial_no_chemotherapy.NonSpatialFusion import ModelRun as ModelRunNotDiverse
from non_spatial_no_chemotherapy.parametrization import ModelParameters
from output import OUTPUT_PATH


baseline_params = ModelParameters(
    number_of_genes=50,
    carrying_capacity=1000,
    number_of_generations=10000,
    mutation_rate_per_gene=0.001,
    fusion_rate=0.01,
    growth_rate=0.5,
    death_rate=0.5,
    save_path=Path(OUTPUT_PATH),
)

result = ModelRunNotDiverse(parameters=baseline_params)
