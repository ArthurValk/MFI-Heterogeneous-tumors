"""Parametrization for the non-spatial models"""

from dataclasses import dataclass
from pathlib import Path


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
    dt: int = 1
    diversity: int | None = None
    seed: int | None = None

    # treatment controls (defaults AFTER required fields)
    treatment_every: int | None = None  # e.g. 20 (off-treatment length)
    treatment_duration: int = 0  # e.g. 20 (on-treatment length)

    treatment_base_extra_death: float = 0.3  # flat extra death during treatment
    treatment_selection: float = 0.1  # fraction of genes that confer resistance (10%)
    treatment_cell_density_dependence: float = (
        0.0  # scaling factor for cell density effects on treatment efficacy
    )


@dataclass(frozen=True)
class ModelResult:
    """Result of the experiment. As model saves files to .csv,
    we return an object referencing these files

    Attributes
    ----------
    metrics_path: Path
        Path to the .csv file containing all metrics
        (Time, TotalCells, NumGenotypes, ShannonIndex, SimpsonIndex, MaxMutations)
    lineage_path: Path
        Path to the .csv file containing the lineage data
    genotype_path: Path
        Path to the .csv file containing the genotype data
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

    # lineage
    cell_count = "TotalCells"
    genotype_id = "GenotypeId"
    ancestor_id = "AncestorId"
