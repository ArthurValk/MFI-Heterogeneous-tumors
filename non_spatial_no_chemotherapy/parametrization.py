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
    lineage_path: Path
        Path to the .csv file containing the lineage data
    mutation_number_path: Path
        Path to the .csv file containing the mutation number data
    score_path: Path
        Path to the .csv file containing the score data
    shannon_path: Path
        Path to the .csv file containing the shannon index data
    simpson_path: Path
        Path to the .csv file containing the simpson index data
    total_data_path: Path
        Path to the .csv file containing the total data
    """

    lineage_path: Path
    mutation_number_path: Path
    score_path: Path
    shannon_path: Path
    simpson_path: Path
    total_data_path: Path
