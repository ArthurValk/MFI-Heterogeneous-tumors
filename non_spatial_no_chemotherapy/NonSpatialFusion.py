"""Modelling heterogeneous tumor growth in a non-spatial setting with fusion between cells."""

from pathlib import Path

import numpy as np

from numba import njit
import time
import sys
import os
import csv

from non_spatial_no_chemotherapy.parametrization import ModelParameters, ModelResult


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
        countPop = countPop + ListePopulation[i]["count"]
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
        if ListePopulation[i]["count"] > 0:
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
        Tot = Tot + ListePopulation[i]["count"]

    for i in range(0, len(ListePopulation)):
        if ListePopulation[i]["count"] > 0:
            prop = float(ListePopulation[i]["count"]) / Tot
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
            Tot = Tot + ListePopulation[i]["count"]

        for i in range(0, len(ListePopulation)):
            prop = float(ListePopulation[i]["count"]) / Tot
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
        if ListePopulation[count]["count"] > 0:
            countPop2 = countPop2 + ListePopulation[count]["count"]
        if y <= countPop2 and ListePopulation[count]["count"] > 0:
            found = 1
            res = count
        count = count + 1
    if found == 0:
        print("Element not found!!!!!")
    return res


def cleanData(ListePopulation):
    """
    Filter population array to keep only genotypes with living cells.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data

    Returns
    -------
    np.ndarray
        Filtered array containing only genotypes with count > 0.
        The cells_to_mutate and fusion_count fields are reset to 0
        for all remaining entries
    """
    # Keep only entries where count > 0
    mask = ListePopulation["count"] > 0
    filtered = ListePopulation[mask].copy()

    # Reset cells_to_mutate and fusion_count for remaining entries
    filtered["cells_to_mutate"] = 0
    filtered["fusion_count"] = 0

    return filtered


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
    hybrid1 = np.zeros(len(G1))
    hybrid2 = np.zeros(len(G1))

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
def MaxScore(ListePopulation):
    """
    Find the maximum number of mutations across all genotypes.

    Parameters
    ----------
    ListePopulation : np.ndarray
        Structured array containing population data with 'genotype' field

    Returns
    -------
    int
        Maximum number of mutations (score) among all genotypes
    """
    res = 0
    for i in range(0, len(ListePopulation)):
        scorei = np.sum(ListePopulation[i]["genotype"])
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
    diversity = parameters.diversity
    # Compute directory path from save_path and parameters
    directory_path = parameters.save_path / f"g{growthRate}" / f"mu{pm}" / f"pf{pf}"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if s:
        np.random.seed(int(s))

    result = _ModelRun(
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
        lineage_path=directory_path / f"Lineage{str(s)}.csv",
        mutation_number_path=directory_path / f"NumberMut{str(s)}.csv",
        score_path=directory_path / f"Score{str(s)}.csv",
        shannon_path=directory_path / f"Shanon{str(s)}.csv",
        simpson_path=directory_path / f"Simpson{str(s)}.csv",
        total_data_path=directory_path / f"Total{str(s)}.csv",
    )
    _write_csv_file(
        model_result.lineage_path,
        ["Time", "CellCount", "GenotypeId", "AncestorId"],
        result["lineage"],
    )
    _write_csv_file(
        model_result.total_data_path,
        ["Time", "TotalCells"],
        result["total_cells"],
    )
    _write_csv_file(
        model_result.mutation_number_path,
        ["Time", "NumGenotypes"],
        result["number_mut"],
    )
    _write_csv_file(
        model_result.shannon_path,
        ["Time", "ShannonIndex"],
        result["shanon"],
    )
    _write_csv_file(
        model_result.simpson_path,
        ["Time", "SimpsonIndex"],
        result["simpson"],
    )
    _write_csv_file(model_result.score_path, ["Time", "MaxMutations"], result["score"])
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
def _ModelRun(
    Ngenes: int,
    NgenerationsMax: int,
    DT: int,
    KC: int,
    pm: float,
    pf: float,
    growthRate: float,
    deathRate: float,
    diversity: int | None = None,
) -> dict:
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
    dict[str, list[str]]
        Dictionary containing simulation results with keys:
        - 'lineage': Cell counts, genotypes, times, and ancestor IDs for extinct and living genotypes
        - 'total_cells': Total population size at each time step
        - 'number_mut': Number of distinct genotypes at each time step
        - 'shanon': Shannon diversity index at each time step
        - 'simpson': Simpson diversity index at each time step
        - 'score': Maximum number of mutations at each time step

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
    Genotype = np.zeros(Ngenes, dtype=bool)

    Number = np.zeros(NgenerationsMax)  # Number of genotypes per time step
    Shanon = np.zeros(NgenerationsMax)  # Shanon index per time step
    Simpson = np.zeros(NgenerationsMax)  # Simpson index per time step
    Score = np.zeros(NgenerationsMax)  # Maximal amount of mutations per time step
    TotalCells = np.zeros(NgenerationsMax)  # Total number of cells per time step

    # Define structured array dtype for population
    pop_dtype = np.dtype(
        [
            ("count", np.int32),
            ("genotype", np.uint8, (Ngenes,)),
            ("cells_to_mutate", np.int32),
            ("birth_time", np.float64),
            ("ancestor_id", np.int32),
            ("fusion_count", np.int32),
            ("genotype_id", np.int32),
        ]
    )

    # Initialize population arrays
    ListePop = np.array([], dtype=pop_dtype)
    ListeExtincted = np.array([], dtype=pop_dtype)
    genotypesCounts = 0

    # Add initial genotype
    initial_entry = np.array(
        [(1, Genotype.copy(), 0, 0, genotypesCounts, 0, genotypesCounts)],
        dtype=pop_dtype,
    )
    ListePop = np.concatenate([ListePop, initial_entry])

    # Add initial diverse genotypes if specified
    if diversity and diversity > 1:
        for i in range(1, diversity):
            genotypesCounts = genotypesCounts + 1
            mutant_genotype = TargetedMutation(Genotype, (i - 1) % Ngenes)
            diverse_entry = np.array(
                [(1, mutant_genotype, 0, 0, 0, 0, genotypesCounts)],
                dtype=pop_dtype,
            )
            ListePop = np.concatenate([ListePop, diverse_entry])

    # Data collection for output files
    lineage_data = []  # f1 data
    total_cells_data = []  # f2 data
    number_mut_data = []  # f3 data
    shanon_data = []  # f4 data
    simpson_data = []  # f5 data
    score_data = []  # f6 data

    # Time loop
    for l in range(0, NgenerationsMax):
        print("Generation=", l)
        TotalPopulation = countPopulation(ListePop)

        # Population loop
        for j in range(0, len(ListePop)):
            nombreRepresentants = ListePop[j]["count"]
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
                    if ListePop[j]["count"] + newCells - newM - newD1 > 0:
                        ListePop[j]["count"] = (
                            ListePop[j]["count"] + newCells - newM - newD1
                        )
                    else:
                        ListePop[j]["count"] = 0
                    if newM - newD2 > 0:
                        ListePop[j]["cells_to_mutate"] = newM - newD2
                    else:
                        ListePop[j]["cells_to_mutate"] = 0
                else:
                    ListePop[j]["count"] = 0
                    ListePop[j]["cells_to_mutate"] = 0

                if ListePop[j]["count"] == 0:
                    NewElement = (
                        1,
                        ListePop[j]["genotype"],
                        ListePop[j]["cells_to_mutate"],
                        l * DT,
                        ListePop[j]["ancestor_id"],
                        ListePop[j]["fusion_count"],
                        ListePop[j]["genotype_id"],
                    )
                    ListeExtincted = add_population_entry(ListeExtincted, NewElement)

        j = 0
        Ncurrent = len(ListePop)
        while j < Ncurrent:
            for k in range(0, ListePop[j]["cells_to_mutate"]):
                GenotypeTemporaire = Mutation(ListePop[j]["genotype"])
                exist = 0
                count = 0
                while count < len(ListePop) and exist == 0:
                    if (
                        convertBinaire(ListePop[count]["genotype"])
                        - convertBinaire(GenotypeTemporaire)
                        == 0
                    ):
                        ListePop[count]["count"] = ListePop[count]["count"] + 1
                        exist = 1
                    count = count + 1
                if exist == 0:
                    genotypesCounts = genotypesCounts + 1
                    NewElement = (
                        1,
                        GenotypeTemporaire,
                        0,
                        l * DT,
                        ListePop[j]["genotype_id"],
                        0,
                        genotypesCounts,
                    )
                    ListePop = add_population_entry(ListePop, NewElement)
            ListePop[j]["cells_to_mutate"] = 0
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
                if ListePop[count]["count"] > 0:
                    countPop2 = countPop2 + ListePop[count]["count"]
                    if y <= countPop2:
                        found = 1
                        ListePop[count]["fusion_count"] = (
                            ListePop[count]["fusion_count"] + 1
                        )
                        ListePop[count]["count"] = ListePop[count]["count"] - 1
                        TotalPopulation = TotalPopulation - 1

                    if ListePop[count]["count"] == 0:
                        NewElement = (
                            1,
                            ListePop[count]["genotype"],
                            ListePop[count]["cells_to_mutate"],
                            l * DT,
                            ListePop[count]["ancestor_id"],
                            ListePop[count]["fusion_count"],
                            ListePop[count]["genotype_id"],
                        )
                        ListeExtincted = add_population_entry(
                            ListeExtincted, NewElement
                        )
                count = count + 1

        Ncurrent = len(ListePop)
        TotalPopulation = countPopulation(ListePop)

        if TotalPopulation > 0:
            for j in range(0, Ncurrent):
                for k in range(0, ListePop[j]["fusion_count"]):
                    Genotype1 = ListePop[j]["genotype"]

                    neighbor = chooseElement(ListePop, TotalPopulation)

                    # Decrement both initiator and neighbor (both consumed in fusion)
                    ListePop[j]["count"] = ListePop[j]["count"] - 1
                    ListePop[neighbor]["count"] = ListePop[neighbor]["count"] - 1
                    TotalPopulation = TotalPopulation - 2

                    # Track extinction of initiator if count reaches zero
                    if ListePop[j]["count"] == 0:
                        NewElement = (
                            1,
                            ListePop[j]["genotype"],
                            ListePop[j]["cells_to_mutate"],
                            l * DT,
                            ListePop[j]["ancestor_id"],
                            ListePop[j]["fusion_count"],
                            ListePop[j]["genotype_id"],
                        )
                        ListeExtincted = add_population_entry(
                            ListeExtincted, NewElement
                        )

                    if ListePop[neighbor]["count"] == 0:
                        NewElement = (
                            1,
                            ListePop[neighbor]["genotype"],
                            ListePop[neighbor]["cells_to_mutate"],
                            l * DT,
                            ListePop[neighbor]["ancestor_id"],
                            ListePop[neighbor]["fusion_count"],
                            ListePop[neighbor]["genotype_id"],
                        )
                        ListeExtincted = add_population_entry(
                            ListeExtincted, NewElement
                        )

                    Genotype2 = ListePop[neighbor]["genotype"]

                    # Two offspring from fusion with recombination (parasexual cycle)
                    hybrid1, hybrid2 = fusionVect(Genotype1, Genotype2)

                    # Add first hybrid
                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if (
                            convertBinaire(ListePop[count]["genotype"])
                            - convertBinaire(hybrid1)
                            == 0
                        ):
                            ListePop[count]["count"] = ListePop[count]["count"] + 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1

                        NewElement = (
                            1,
                            hybrid1,
                            0,
                            l * DT,
                            ListePop[j]["genotype_id"],
                            0,
                            genotypesCounts,
                        )  # New hybrid genotype!
                        ListePop = add_population_entry(ListePop, NewElement)

                    # Add second hybrid
                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if (
                            convertBinaire(ListePop[count]["genotype"])
                            - convertBinaire(hybrid2)
                            == 0
                        ):
                            ListePop[count]["count"] = ListePop[count]["count"] + 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1

                        NewElement = (
                            1,
                            hybrid2,
                            0,
                            l * DT,
                            ListePop[j]["genotype_id"],
                            0,
                            genotypesCounts,
                        )  # New hybrid genotype!
                        ListePop = add_population_entry(ListePop, NewElement)

                    # Net population change: -2 (both parents) +2 (two hybrids) = 0 per fusion
                    TotalPopulation = TotalPopulation + 2

                ListePop[j]["fusion_count"] = 0

        else:
            for j in range(0, Ncurrent):
                for k in range(0, ListePop[j]["fusion_count"]):
                    ListePop[j]["count"] = ListePop[j]["count"] + 1

        ListePop = cleanData(ListePop)

        Number[l] = countSpecies(ListePop)
        TotalCells[l] = countPopulation(ListePop)
        Shanon[l] = ComputeShanon(ListePop)
        Simpson[l] = ComputeIndex(ListePop, 2)
        Score[l] = MaxScore(ListePop)

        # Collecting results for output files
        if l % 50 == 0 or l == NgenerationsMax - 1:
            for i in range(0, len(ListeExtincted)):
                lineage_data.append(
                    [
                        l * DT,
                        1,
                        ListeExtincted[i]["genotype_id"],
                        ListeExtincted[i]["ancestor_id"],
                    ]
                )

            for i in range(0, len(ListePop)):
                lineage_data.append(
                    [
                        l * DT,
                        ListePop[i]["count"],
                        ListePop[i]["genotype_id"],
                        ListePop[i]["ancestor_id"],
                    ]
                )

            total_cells_data.append([l * DT, TotalCells[l]])
            number_mut_data.append([l * DT, Number[l]])
            shanon_data.append([l * DT, Shanon[l]])
            simpson_data.append([l * DT, Simpson[l]])
            score_data.append([l * DT, Score[l]])

            ListeExtincted = np.array([], dtype=ListeExtincted.dtype)

    # Return all collected data for file writing
    return {
        "lineage": lineage_data,
        "total_cells": total_cells_data,
        "number_mut": number_mut_data,
        "shanon": shanon_data,
        "simpson": simpson_data,
        "score": score_data,
    }


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
