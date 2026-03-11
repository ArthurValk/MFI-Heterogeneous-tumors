"""Parametrization for the non-spatial models"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelParameters:
    """Parameters of the experiment"""

    # required (no defaults) FIRST
    number_of_genes: int
    carrying_capacity: int
    number_of_generations: int
    mutation_rate_per_gene: float
    fusion_rate: float
    growth_rate: float
    death_rate: float
    save_path: Path

    # defaults AFTER
    dt: float = 0.25  # hours; default = 15 minutes
    data_resolution: int = 4  # store every 4 steps = every hour if dt=0.25
    diversity: int | None = None
    seed: int | None = None
    initial_population_size: int = 1

    # treatment controls
    treatment_injection_every: int | None = None  # in simulation steps
    treatment_initial_concentration: float = 0.25
    treatment_halflife: float = 12.0  # hours
    treatment_concentration_to_extra_death: float = (
        0.7 / 48.0
    )  # per 15 min by default if original was 0.7 per 12h

    # resistance controls
    treatment_selection: float = 0.1
    treatment_resistivity: float = 1.0


@dataclass(frozen=True)
class ModelResult:
    """Result of the experiment. As model saves files to .csv,
    we return an object referencing these files
    """

    metrics_path: Path
    lineage_path: Path
    genotype_path: Path


class MetricNames:
    """Names of the metrics"""

    # metrics
    time = "Time"
    total_cells = "TotalCells"
    num_genotypes = "NumGenotypes"
    shannon_index = "ShannonIndex"
    simpson_index = "SimpsonIndex"
    max_mutations = "MaxMutations"
    drug_concentration = "DrugConcentration"
    drug_extra_death_wt = "DrugExtraDeathWT"

    # lineage
    cell_count = "TotalCells"
    genotype_id = "GenotypeId"
    ancestor_id = "AncestorId"

    # monte carlo specific
    seed = "seed"
    births = "births"
    deaths = "deaths"
    fusions = "fusions"
    mutations = "mutations"
    genotype_array = "genotype_array"