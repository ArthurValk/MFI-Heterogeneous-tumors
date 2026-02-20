"""Modelling heterogeneous tumor growth in a non-spatial setting with fusion between cells."""

from pathlib import Path

import numpy as np

from numba import njit
import time
import sys
import os
import csv

from non_spatial_no_chemotherapy.parametrization import (
    ModelParameters,
    ModelResult,
    MetricNames,
)


# Field indices for population data tuples (used in _ModelRun with @njit)
POP_COUNT = 0
POP_GENOTYPE = 1
POP_CELLS_TO_MUTATE = 2
POP_BIRTH_TIME = 3
POP_ANCESTOR_ID = 4
POP_FUSION_COUNT = 5
POP_GENOTYPE_ID = 6

DATA_RESOLUTION = 25


# Functions
@njit
def countPopulation(ListePopulation):
    """
    Count the total number of cells across all genotypes.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data with 'count' field
        representing the number of cells for each genotype

    Returns
    -------
    int
        Total number of cells summed across all genotypes
    """
    countPop = 0
    for i in range(0, len(ListePopulation)):
        countPop = countPop + ListePopulation[i][POP_COUNT]
    return countPop


@njit
def countSpecies(ListePopulation):
    """
    Count the number of distinct genotypes with at least one cell.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data

    Returns
    -------
    int
        Number of genotypes with count > 0
    """
    countS = 0
    for i in range(0, len(ListePopulation)):
        if ListePopulation[i][POP_COUNT] > 0:
            countS = countS + 1
    return countS


@njit
def convertBinaire(x):
    """
    Convert a binary array to an integer representation.

    Parameters
    ----------
    x : np.ndarray
        Binary array (values 0 or 1), where index i corresponds to 2^i

    Returns
    -------
    int
        Integer representation of the binary array
    """
    n = len(x)
    res = 0
    for i in range(0, n):
        res = res + x[i] * (2**i)
    return res


@njit
def ComputeShanon(ListePopulation):
    """
    Compute the Shannon diversity index for the population.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data with 'count' field

    Returns
    -------
    float
        Shannon entropy index: -sum(p_i * log(p_i)) where p_i is the
        proportion of cells with genotype i
    """
    res = 0
    Tot = 0
    for i in range(0, len(ListePopulation)):
        Tot = Tot + ListePopulation[i][POP_COUNT]

    for i in range(0, len(ListePopulation)):
        if ListePopulation[i][POP_COUNT] > 0:
            prop = float(ListePopulation[i][POP_COUNT]) / Tot
            res = res - prop * np.log(prop)

    return res


@njit
def ComputeIndex(ListePopulation, q):
    """
    Compute the generalized Renyi diversity index of order q.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data with 'count' field
    q : int
        Order of the diversity index. q=1 computes Shannon entropy,
        q=2 computes Simpson index

    Returns
    -------
    float
        Renyi diversity index of order q
    """
    res = 0
    if q == 1:
        res = ComputeShanon(ListePopulation)
    else:
        Tot = 0
        for i in range(0, len(ListePopulation)):
            Tot = Tot + ListePopulation[i][POP_COUNT]

        for i in range(0, len(ListePopulation)):
            prop = float(ListePopulation[i][POP_COUNT]) / Tot
            res = res + prop**q
        res = res ** (float(1) / (1 - q))

    return res


@njit
def Mutation(x):
    """
    Generate a mutated genotype by flipping one zero to one.

    Parameters
    ----------
    x : np.ndarray
        Binary genotype array where 1 indicates a mutation, 0 indicates wild-type

    Returns
    -------
    np.ndarray
        Mutated genotype with a locus set to 1. If available unmutated loci exist,
        one is randomly selected. If all loci are mutated, randomly picks any locus
        and sets it to 1 (trivial)
    """
    res = x.copy()

    # Get all loci indices and filter to available (unmutated) loci
    all_loci = np.arange(len(x))
    available_loci = all_loci[res == 0] if np.any(res == 0) else all_loci

    # Randomly mutate one available locus (harmless if setting 1 to 1)
    locus = available_loci[np.random.randint(0, len(available_loci))]
    res[locus] = 1

    return res


@njit
def TargetedMutation(x, j):
    """
    Generate a genotype with a mutation at a specific locus.

    Parameters
    ----------
    x : np.ndarray
        Binary genotype array
    j : int
        Locus index to mutate (set to 1)

    Returns
    -------
    np.ndarray
        Genotype with mutation at locus j. Used for initializing diverse starting populations.
    """
    res = x.copy()
    if j < len(res):
        res[j] = 1
    return res


@njit
def chooseElement(ListePopulation, total_pop):
    """
    Randomly select a genotype index weighted by cell count.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data with 'count' field
    total_pop : int
        Total number of cells in the population

    Returns
    -------
    int
        Index of selected genotype, chosen proportionally to its population size
    """
    res = 0

    if total_pop == 1:
        y = 1
    else:
        y = np.random.randint(1, total_pop + 1)
    countPop2 = 0
    count = 0
    found = 0
    while count < len(ListePopulation) and found == 0:
        if ListePopulation[count][POP_COUNT] > 0:
            countPop2 = countPop2 + ListePopulation[count][POP_COUNT]
        if y <= countPop2 and ListePopulation[count][POP_COUNT] > 0:
            found = 1
            res = count
        count = count + 1
    if found == 0:
        print("Element not found!!!!!")
    return res


@njit
def cleanData(ListePopulation):
    """
    Filter population list to keep only genotypes with living cells.

    Parameters
    ----------
    ListePopulation : list
        List of population entries (as lists with integer indices)

    Returns
    -------
    list
        Filtered list containing only genotypes with count > 0.
        The cells_to_mutate and fusion_count fields are reset to 0
        for all remaining entries
    """
    # Filter and reset
    result = []
    for entry in ListePopulation:
        if entry[POP_COUNT] > 0:
            new_entry = list(entry)  # Copy the entry
            new_entry[POP_CELLS_TO_MUTATE] = 0
            new_entry[POP_FUSION_COUNT] = 0
            result.append(new_entry)

    return result


@njit
def fusionVect(G1, G2):
    """
    Perform genetic recombination between two fused genotypes.

    Models the parasexual cycle: fusion creates a polyploid cell that then undergoes
    ploidy reduction, producing two distinct haploid offspring with segregation of
    parental genetic material.

    Parameters
    ----------
    G1 : np.ndarray
        First parent genotype
    G2 : np.ndarray
        Second parent genotype

    Returns
    -------
    tuple of np.ndarray
        Two hybrid offspring genotypes resulting from fusion recombination.
        At each locus, offspring complementarily inherit from parents: one gets the allele
        from G1 while the other gets from G2, representing proper genetic segregation
        during ploidy reduction.
    """
    hybrid1 = np.zeros(len(G1), dtype=np.bool_)
    hybrid2 = np.zeros(len(G1), dtype=np.bool_)

    for i in range(0, len(G1)):
        # Randomly decide which offspring gets which parent's allele
        # Ensures proper segregation: one offspring gets G1[i], other gets G2[i]
        y = np.random.randint(0, 2)
        if y == 0:
            hybrid1[i] = G1[i]
            hybrid2[i] = G2[i]
        else:
            hybrid1[i] = G2[i]
            hybrid2[i] = G1[i]

    return hybrid1, hybrid2


@njit
def MaxScore(ListePopulation, all_genotypes):
    """
    Find the maximum number of mutations across all genotypes.

    Parameters
    ----------
    ListePopulation : list
        List of population entries (with genotype indices at POP_GENOTYPE)
    all_genotypes : list
        List of all genotype arrays

    Returns
    -------
    int
        Maximum number of mutations (score) among all genotypes
    """
    res = 0
    for i in range(0, len(ListePopulation)):
        genotype_idx = ListePopulation[i][POP_GENOTYPE]
        scorei = np.sum(all_genotypes[genotype_idx])
        res = scorei if res < scorei else res
    return res


def _write_csv_file(filepath: Path, header: list[str], data: list[list]) -> None:
    """
    Write data to a CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the output CSV file
    header : list[str]
        Header row with column names
    data : list[list]
        List of data rows, where each row is a list of values
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def ModelRun(parameters: ModelParameters) -> ModelResult:
    """
    Main entry point for running the heterogeneous tumor growth model.

    Parameters
    ----------
    parameters : ModelParameters
        Configuration object containing:
        - number_of_genes: Number of possible mutations
        - number_of_generations: Number of time steps to simulate
        - dt: Time step size (default 1)
        - seed: Random seed for reproducibility
        - carrying_capacity: Maximum population size (KC)
        - mutation_rate_per_gene: Probability of mutation per gene
        - fusion_rate: Rate of cell fusion events
        - growth_rate: Birth rate of cells
        - death_rate: Death rate of cells
        - save_path: Directory to save output files

    Returns
    -------
    ModelResult
        Object containing paths to the generated CSV files with simulation results:
        - metrics_path: Path to metrics CSV (time, total_cells, num_genotypes, shannon_index, simpson_index, max_mutations)
        - lineage_path: Path to lineage CSV (time, cell_count, genotype_id, ancestor_id)
        - genotype_path: Path to genotypes CSV (genotype_id, locus_0, locus_1, ..., locus_N)
    """
    Ngenes = parameters.number_of_genes
    NgenerationsMax = parameters.number_of_generations
    DT = parameters.dt
    s = parameters.seed
    KC = parameters.carrying_capacity
    pm = parameters.mutation_rate_per_gene
    pf = parameters.fusion_rate
    growthRate = parameters.growth_rate
    deathRate = parameters.death_rate
    diversity = parameters.diversity if parameters.diversity is not None else 0
    # Compute directory path from save_path and parameters
    directory_path = parameters.save_path / f"g{growthRate}" / f"mu{pm}" / f"pf{pf}"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if s:
        np.random.seed(int(s))

    lineage_data, metrics_data, all_genotypes = _ModelRun(
        Ngenes=Ngenes,
        NgenerationsMax=NgenerationsMax,
        DT=DT,
        KC=KC,
        pm=pm,
        pf=pf,
        growthRate=growthRate,
        deathRate=deathRate,
        diversity=diversity,
    )

    # Write collected data to CSV files
    model_result = ModelResult(
        metrics_path=directory_path / f"Metrics{str(s)}.csv",
        lineage_path=directory_path / f"Lineage{str(s)}.csv",
        genotype_path=directory_path / f"Genotypes{str(s)}.csv",
    )
    _write_csv_file(
        model_result.metrics_path,
        [
            MetricNames.time,
            MetricNames.total_cells,
            MetricNames.num_genotypes,
            MetricNames.shannon_index,
            MetricNames.simpson_index,
            MetricNames.max_mutations,
        ],
        metrics_data,
    )
    _write_csv_file(
        model_result.lineage_path,
        [
            MetricNames.time,
            MetricNames.cell_count,
            MetricNames.genotype_id,
            MetricNames.ancestor_id,
        ],
        lineage_data,
    )
    _write_csv_file(
        model_result.genotype_path,
        ["GenotypeId"] + [f"Locus{i}" for i in range(Ngenes)],
        [[idx] + genotype.tolist() for idx, genotype in enumerate(all_genotypes)],
    )
    return model_result


# Helper function for structured array operations
def add_population_entry(arr, entry_tuple):
    """
    Add a new entry to a population structured array.

    Parameters
    ----------
    arr : np.ndarray
        Structured array containing population data
    entry_tuple : tuple
        New entry to add, containing values for all fields
        (count, genotype, cells_to_mutate, birth_time, ancestor_id, fusion_count, genotype_id)

    Returns
    -------
    np.ndarray
        New structured array with the entry appended
    """
    # Make a copy of genotype if it's an array
    entry_list = list(entry_tuple)
    if len(entry_list) > 1 and hasattr(entry_list[1], "copy"):
        entry_list[1] = entry_list[1].copy()
    new_entry = np.array([tuple(entry_list)], dtype=arr.dtype)
    return np.concatenate([arr, new_entry])


# Main Function
@njit
def _ModelRun(
    Ngenes: int,
    NgenerationsMax: int,
    DT: int,
    KC: int,
    pm: float,
    pf: float,
    growthRate: float,
    deathRate: float,
    diversity: int = 0,
):
    """
    Core model simulation for heterogeneous tumor growth with cell fusion.

    Simulates a non-spatial, discrete-time stochastic model of tumor evolution
    where cells can undergo mutations, divide, die, and fuse. Genotypes are
    represented as binary arrays of mutations, and population dynamics follow
    logistic growth with density-dependent death.

    Parameters
    ----------
    Ngenes : int
        Number of possible genes/mutations (determines genotype size)
    NgenerationsMax : int
        Number of discrete time steps to simulate
    DT : int
        Time step increment (typically 1)
    KC : int
        Carrying capacity of the population (determines competition strength)
    pm : float
        Mutation rate per gene per cell division event
    pf : float
        Fusion rate (probability of cell fusion event)
    growthRate : float
        Birth rate parameter (lambda in logistic growth)
    deathRate : float
        Death rate parameter (mu in logistic growth)
    diversity : int | None, optional
        Number of initial genotypes to seed (default: None, only wildtype)
        If diversity > 1, starts with wildtype + (diversity-1) genotypes with single mutations
        at different loci (0, 1, 2, ..., diversity-2)

    Returns
    -------
    tuple[list, list, list]
        lineage_data: List of lists containing lineage information for each time step
        metrics_data: List of lists containing population metrics for each time step
        all_genotypes: List of all unique genotype arrays encountered during the simulation

    Notes
    -----
    The simulation proceeds through the following phases at each time step:

    1. **Births and Deaths**: For each genotype:
       - Generate new cells via Poisson process: births ~ Poisson(growthRate * count * DT)
       - Generate mutants from births via binomial: mutants ~ Binomial(births, Ngenes * pm)
       - Generate deaths via Poisson: deaths ~ Poisson(deathRate * count * totalPop * DT / KC)

    2. **Mutation Processing**: Process mutant cells, creating new genotypes

    3. **Fusion Events**: Randomly select cells for fusion events:
       - Generate new fusions ~ Poisson(pf * totalPop * DT)
       - For each fusion, recombine genotypes via fusionVect()

    4. **Data Collection**: Record lineage, population statistics, and diversity indices
    """
    Genotype = np.zeros(Ngenes, dtype=np.bool_)

    Number = np.zeros(NgenerationsMax)  # Number of genotypes per time step
    Shanon = np.zeros(NgenerationsMax)  # Shanon index per time step
    Simpson = np.zeros(NgenerationsMax)  # Simpson index per time step
    Score = np.zeros(NgenerationsMax)  # Maximal amount of mutations per time step
    TotalCells = np.zeros(NgenerationsMax)  # Total number of cells per time step

    # Initialize population arrays and genotype storage
    all_genotypes = []  # Store all unique genotype arrays
    ListePop = []  # Store population entries with genotype indices
    ListeExtincted = []
    genotypesCounts = 0

    # Add initial genotype
    all_genotypes.append(Genotype.copy())
    initial_entry = [1, 0, 0, 0, genotypesCounts, 0, genotypesCounts]  # genotype_idx=0
    ListePop.append(initial_entry)

    # Add initial diverse genotypes if specified
    if diversity > 1:
        for i in range(1, diversity):
            genotypesCounts = genotypesCounts + 1
            mutant_genotype = TargetedMutation(Genotype, (i - 1) % Ngenes)
            all_genotypes.append(mutant_genotype)
            diverse_entry = [1, len(all_genotypes) - 1, 0, 0, 0, 0, genotypesCounts]
            ListePop.append(diverse_entry)

    # Data collection for output files
    lineage_data = []  # f1 data
    metrics_data = []  # consolidated metrics data

    # Time loop
    for l in range(0, NgenerationsMax):
        print("Generation=", l)
        TotalPopulation = countPopulation(ListePop)

        # Population loop
        for j in range(0, len(ListePop)):
            nombreRepresentants = ListePop[j][POP_COUNT]
            if nombreRepresentants > 0:
                newCells = np.random.poisson(
                    growthRate * nombreRepresentants * DT
                )  # Newborns
                newM = np.random.binomial(newCells, Ngenes * pm)  # New mutants
                if newM > newCells:
                    newM = newCells

                newD = np.random.poisson(
                    deathRate * nombreRepresentants * TotalPopulation * DT / KC
                )  # New dead cells

                countDeads = 0
                newD1 = 0
                newD2 = 0

                if newD < nombreRepresentants + newCells:
                    while countDeads < newD:
                        m = np.random.randint(1, nombreRepresentants + newCells)
                        if (
                            m < nombreRepresentants + newCells - newM
                            and newD1 < nombreRepresentants + newCells - newM
                        ):
                            newD1 = newD1 + 1
                        else:
                            newD2 = newD2 + 1
                        countDeads = countDeads + 1
                    if ListePop[j][POP_COUNT] + newCells - newM - newD1 > 0:
                        ListePop[j][POP_COUNT] = (
                            ListePop[j][POP_COUNT] + newCells - newM - newD1
                        )
                    else:
                        ListePop[j][POP_COUNT] = 0
                    if newM - newD2 > 0:
                        ListePop[j][POP_CELLS_TO_MUTATE] = newM - newD2
                    else:
                        ListePop[j][POP_CELLS_TO_MUTATE] = 0
                else:
                    ListePop[j][POP_COUNT] = 0
                    ListePop[j][POP_CELLS_TO_MUTATE] = 0

                if ListePop[j][POP_COUNT] == 0:
                    extinct_entry = [
                        1,
                        ListePop[j][POP_GENOTYPE],
                        ListePop[j][POP_CELLS_TO_MUTATE],
                        l * DT,
                        ListePop[j][POP_ANCESTOR_ID],
                        ListePop[j][POP_FUSION_COUNT],
                        ListePop[j][POP_GENOTYPE_ID],
                    ]
                    ListeExtincted.append(extinct_entry)

        j = 0
        Ncurrent = len(ListePop)
        while j < Ncurrent:
            for k in range(0, ListePop[j][POP_CELLS_TO_MUTATE]):
                parent_genotype = all_genotypes[ListePop[j][POP_GENOTYPE]]
                GenotypeTemporaire = Mutation(parent_genotype)
                exist = 0
                count = 0
                while count < len(ListePop) and exist == 0:
                    if (
                        convertBinaire(all_genotypes[ListePop[count][POP_GENOTYPE]])
                        - convertBinaire(GenotypeTemporaire)
                        == 0
                    ):
                        ListePop[count][POP_COUNT] = ListePop[count][POP_COUNT] + 1
                        exist = 1
                    count = count + 1
                if exist == 0:
                    genotypesCounts = genotypesCounts + 1
                    all_genotypes.append(GenotypeTemporaire)
                    new_genotype_entry = [
                        1,
                        len(all_genotypes) - 1,  # genotype index
                        0,
                        l * DT,
                        ListePop[j][POP_GENOTYPE_ID],
                        0,
                        genotypesCounts,
                    ]
                    ListePop.append(new_genotype_entry)
            ListePop[j][POP_CELLS_TO_MUTATE] = 0
            j = j + 1

        # Hybrids formation
        TotalPopulation = countPopulation(ListePop)
        newH = np.random.poisson(pf * TotalPopulation * DT)  # New hybrids
        newH = int(np.minimum(newH, int(TotalPopulation / 2)))

        for j in range(0, newH):
            y = np.random.randint(1, TotalPopulation + 1)
            countPop2 = 0
            count = 0
            found = 0

            while count < len(ListePop) and found == 0:
                if ListePop[count][POP_COUNT] > 0:
                    countPop2 = countPop2 + ListePop[count][POP_COUNT]
                    if y <= countPop2:
                        found = 1
                        ListePop[count][POP_FUSION_COUNT] = (
                            ListePop[count][POP_FUSION_COUNT] + 1
                        )
                        ListePop[count][POP_COUNT] = ListePop[count][POP_COUNT] - 1
                        TotalPopulation = TotalPopulation - 1

                    if ListePop[count][POP_COUNT] == 0:
                        extinct_entry = [
                            1,
                            ListePop[count][POP_GENOTYPE],
                            ListePop[count][POP_CELLS_TO_MUTATE],
                            l * DT,
                            ListePop[count][POP_ANCESTOR_ID],
                            ListePop[count][POP_FUSION_COUNT],
                            ListePop[count][POP_GENOTYPE_ID],
                        ]
                        ListeExtincted.append(extinct_entry)
                count = count + 1

        Ncurrent = len(ListePop)
        TotalPopulation = countPopulation(ListePop)

        if TotalPopulation > 0:
            for j in range(0, Ncurrent):
                for k in range(0, ListePop[j][POP_FUSION_COUNT]):
                    Genotype1 = all_genotypes[ListePop[j][POP_GENOTYPE]]

                    neighbor = chooseElement(ListePop, TotalPopulation)

                    # Decrement both initiator and neighbor (both consumed in fusion)
                    ListePop[j][POP_COUNT] = ListePop[j][POP_COUNT] - 1
                    ListePop[neighbor][POP_COUNT] = ListePop[neighbor][POP_COUNT] - 1
                    TotalPopulation = TotalPopulation - 2

                    # Track extinction of initiator if count reaches zero
                    if ListePop[j][POP_COUNT] == 0:
                        extinct_entry = [
                            1,
                            ListePop[j][POP_GENOTYPE],
                            ListePop[j][POP_CELLS_TO_MUTATE],
                            l * DT,
                            ListePop[j][POP_ANCESTOR_ID],
                            ListePop[j][POP_FUSION_COUNT],
                            ListePop[j][POP_GENOTYPE_ID],
                        ]
                        ListeExtincted.append(extinct_entry)

                    if ListePop[neighbor][POP_COUNT] == 0:
                        extinct_entry = [
                            1,
                            ListePop[neighbor][POP_GENOTYPE],
                            ListePop[neighbor][POP_CELLS_TO_MUTATE],
                            l * DT,
                            ListePop[neighbor][POP_ANCESTOR_ID],
                            ListePop[neighbor][POP_FUSION_COUNT],
                            ListePop[neighbor][POP_GENOTYPE_ID],
                        ]
                        ListeExtincted.append(extinct_entry)

                    Genotype2 = all_genotypes[ListePop[neighbor][POP_GENOTYPE]]

                    # Two offspring from fusion with recombination (parasexual cycle)
                    hybrid1, hybrid2 = fusionVect(Genotype1, Genotype2)

                    # Add first hybrid
                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if (
                            convertBinaire(all_genotypes[ListePop[count][POP_GENOTYPE]])
                            - convertBinaire(hybrid1)
                            == 0
                        ):
                            ListePop[count][POP_COUNT] = ListePop[count][POP_COUNT] + 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1
                        all_genotypes.append(hybrid1)
                        hybrid_entry = [
                            1,
                            len(all_genotypes) - 1,  # genotype index
                            0,
                            l * DT,
                            ListePop[j][POP_GENOTYPE_ID],
                            0,
                            genotypesCounts,
                        ]  # New hybrid genotype!
                        ListePop.append(hybrid_entry)

                    # Add second hybrid
                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if (
                            convertBinaire(all_genotypes[ListePop[count][POP_GENOTYPE]])
                            - convertBinaire(hybrid2)
                            == 0
                        ):
                            ListePop[count][POP_COUNT] = ListePop[count][POP_COUNT] + 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1
                        all_genotypes.append(hybrid2)
                        hybrid_entry = [
                            1,
                            len(all_genotypes) - 1,  # genotype index
                            0,
                            l * DT,
                            ListePop[j][POP_GENOTYPE_ID],
                            0,
                            genotypesCounts,
                        ]  # New hybrid genotype!
                        ListePop.append(hybrid_entry)

                    # Net population change: -2 (both parents) +2 (two hybrids) = 0 per fusion
                    TotalPopulation = TotalPopulation + 2

                ListePop[j][POP_FUSION_COUNT] = 0

        else:
            for j in range(0, Ncurrent):
                for k in range(0, ListePop[j][POP_FUSION_COUNT]):
                    ListePop[j][POP_COUNT] = ListePop[j][POP_COUNT] + 1

        ListePop = cleanData(ListePop)

        # Collecting results for output files
        if l % DATA_RESOLUTION == 0 or l == NgenerationsMax - 1:
            Number[l] = countSpecies(ListePop)
            TotalCells[l] = countPopulation(ListePop)
            Shanon[l] = ComputeShanon(ListePop)
            Simpson[l] = ComputeIndex(ListePop, 2)
            Score[l] = MaxScore(ListePop, all_genotypes)
            for i in range(0, len(ListeExtincted)):
                lineage_data.append(
                    [
                        l * DT,
                        1,
                        ListeExtincted[i][POP_GENOTYPE_ID],
                        ListeExtincted[i][POP_ANCESTOR_ID],
                    ]
                )

            for i in range(0, len(ListePop)):
                lineage_data.append(
                    [
                        l * DT,
                        ListePop[i][POP_COUNT],
                        ListePop[i][POP_GENOTYPE_ID],
                        ListePop[i][POP_ANCESTOR_ID],
                    ]
                )

            metrics_data.append(
                [l * DT, TotalCells[l], Number[l], Shanon[l], Simpson[l], Score[l]]
            )

            ListeExtincted = ListeExtincted[:0]  # Clear while preserving type

    # Return all collected data for file writing (as tuple for Numba compatibility)
    return lineage_data, metrics_data, all_genotypes


if __name__ == "main":
    start = time.time()

    # Parse command-line arguments or use defaults
    params = ModelParameters(
        seed=int(sys.argv[1]),
        number_of_genes=int(sys.argv[2]),
        carrying_capacity=int(sys.argv[3]),
        number_of_generations=int(sys.argv[4]),
        mutation_rate_per_gene=float(sys.argv[5]),
        fusion_rate=float(sys.argv[6]),
        growth_rate=float(sys.argv[7]),
        death_rate=float(sys.argv[8]),
        save_path=Path("Results") / f"CC{int(sys.argv[3])}" / "Neutral/LogisticFusion",
        diversity=int(sys.argv[9]),
    )

    ModelRun(parameters=params)

    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))
