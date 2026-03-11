"""File to coordinate the running of experiments for the non-spatial model."""

from non_spatial.NonSpatialFusion import ModelRun
from non_spatial.parametrization import ModelParameters
from output import OUTPUT_PATH

# Time step = 15 minutes = 0.25 hours
# Anything that was "per 12 hours" should be divided by 48 to preserve biology,
# except mutation_rate_per_gene, which is per birth event in this model.

baseline_params = ModelParameters(
    number_of_genes=100,
    carrying_capacity=3000,
    number_of_generations=24 * 4 * 140,  # 140 days, 15-minute steps
    mutation_rate_per_gene=1e-4,         # per birth event -> unchanged
    fusion_rate=1e-4 / 48.0,            # rescaled from per-12h to per-15min
    growth_rate=0.12 / 48.0,            # rescaled from per-12h to per-15min
    death_rate=0.04 / 48.0,             # rescaled from per-12h to per-15min
    save_path=OUTPUT_PATH,
    dt=0.25,                            # 15 minutes = 0.25 hours
    data_resolution=4,                  # store every hour
    diversity=1,
    seed=0,
    treatment_injection_every=21 * 24 * 4,      # every 3 weeks
    treatment_initial_concentration=0.25,
    treatment_halflife=12.0,                    # 12 hours
    treatment_concentration_to_extra_death=0.7 / 48.0,  # per 15 min
    treatment_selection=0.1,
    treatment_resistivity=1.0,
)

result = ModelRun(parameters=baseline_params)
print("Metrics:", result.metrics_path)
print("Lineage:", result.lineage_path)
print("Genotypes:", result.genotype_path)