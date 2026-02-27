"""File to coordinate the running of experiments for the non-spatial model."""

from pathlib import Path

from non_spatial_no_chemotherapy.NonSpatialFusion import ModelRun
from non_spatial_no_chemotherapy.parametrization import ModelParameters
from output import OUTPUT_PATH

baseline_params = ModelParameters(
    number_of_genes=50,
    carrying_capacity=1000,
    number_of_generations=100,
    mutation_rate_per_gene=10**-2,
    fusion_rate=10**-2,
    growth_rate=0.5,
    death_rate=0.5,
    save_path=Path(OUTPUT_PATH),
    treatment_every=20,
    treatment_duration=20,
    treatment_base_extra_death=0.3,
    treatment_selection=0.1,
)

result = ModelRun(parameters=baseline_params)
print("Metrics:", result.metrics_path)
print("Lineage:", result.lineage_path)
print("Genotypes:", result.genotype_path)
