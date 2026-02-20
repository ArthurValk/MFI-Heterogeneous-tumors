"""Parametrization for the non-spatial models"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelParameters:
    """Parameters of the experiment"""

    number_of_genes: int
    carrying_capacity: int
    number_of_generations: int
    mutation_rate_per_gene: float
    fusion_rate: float
    growth_rate: float
    death_rate: float
    save_path: Path
    # stepsize
    dt: int = 1
    # argument only relevant for NonSpatialFusionDiversity.py:
    diversity: int | None = None
    # seed
    seed: int | None = None


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
