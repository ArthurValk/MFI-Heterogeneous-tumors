"""Modelling heterogeneous tumor growth in a non-spatial setting with fusion between cells."""

from pathlib import Path

import numpy as np
from numba.typed import List
from numba import njit
import time
import sys
import os
import csv

try:
    from parametrization import (
        ModelParameters,
        ModelResult,
        MetricNames,
    )
except:
    from non_spatial.parametrization import (
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


@njit
def countPopulation(ListePopulation):
    countPop = 0
    for i in range(0, len(ListePopulation)):
        countPop = countPop + ListePopulation[i][POP_COUNT]
    return countPop


@njit
def countSpecies(ListePopulation):
    countS = 0
    for i in range(0, len(ListePopulation)):
        if ListePopulation[i][POP_COUNT] > 0:
            countS = countS + 1
    return countS


@njit
def ComputeShanon(ListePopulation):
    res = 0.0
    Tot = 0
    for i in range(0, len(ListePopulation)):
        Tot = Tot + ListePopulation[i][POP_COUNT]

    if Tot == 0:
        return 0.0

    for i in range(0, len(ListePopulation)):
        if ListePopulation[i][POP_COUNT] > 0:
            prop = float(ListePopulation[i][POP_COUNT]) / Tot
            res = res - prop * np.log(prop)

    return res


@njit
def ComputeIndex(ListePopulation, q):
    res = 0.0
    if q == 1:
        res = ComputeShanon(ListePopulation)
    else:
        Tot = 0
        for i in range(0, len(ListePopulation)):
            Tot = Tot + ListePopulation[i][POP_COUNT]

        if Tot == 0:
            return 0.0

        for i in range(0, len(ListePopulation)):
            prop = float(ListePopulation[i][POP_COUNT]) / Tot
            res = res + prop**q
        if res <= 0.0:
            return 0.0
        res = res ** (float(1) / (1 - q))

    return res


@njit
def Mutation(x):
    """
    Attempt a mutation at one random locus.
    If locus is 0/False -> set to 1/True.
    If locus is already 1/True -> do nothing.
    """
    res = x.copy()
    locus = np.random.randint(0, len(res))
    if not res[locus]:
        res[locus] = True
    return res


@njit
def TargetedMutation(x, j):
    res = x.copy()
    if j < len(res):
        res[j] = 1
    return res


@njit
def is_injection_time(l: int, every: int) -> bool:
    if every <= 0:
        return False
    return l > 0 and l % every == 0


@njit
def compute_decay_factor(dt_hours: float, halflife_hours: float) -> float:
    """
    Exponential decay factor over one time step:
        c(t + dt) = c(t) * exp(-ln(2) * dt / halflife)
    """
    if halflife_hours <= 0.0:
        return 0.0
    return np.exp(-np.log(2.0) * dt_hours / halflife_hours)


@njit
def make_resistivity(Ngenes: int, selection: float, treatment_resistivity: float):
    """
    Returns a dimensionless resistivity array of length Ngenes.

    Exactly k genes (k = round(selection*Ngenes), at least 1 if selection>0)
    get coefficients drawn uniformly from [0, treatment_resistivity].
    Others are 0.
    """
    resist = np.zeros(Ngenes, dtype=np.float64)

    if selection <= 0.0 or treatment_resistivity <= 0.0:
        return resist

    k = int(selection * Ngenes + 0.5)
    if k < 1:
        k = 1
    if k > Ngenes:
        k = Ngenes

    idx = np.arange(Ngenes)
    np.random.shuffle(idx)
    chosen = idx[:k]

    for i in range(k):
        resist[chosen[i]] = np.random.random() * treatment_resistivity

    return resist


@njit
def genotype_resistance_score(genotype, resistivity) -> float:
    """
    Precompute the total dimensionless resistance score for a genotype:
        score = sum(resistivity[i] for mutated loci)
    """
    s = 0.0
    for i in range(len(genotype)):
        if genotype[i]:
            s += resistivity[i]
    return s


@njit
def genotype_extra_death_from_score(
    resistance_score: float, current_extra_death: float
) -> float:
    """
    Compute genotype-specific treatment extra death from a precomputed
    resistance score and the current WT treatment pressure.
    """
    if current_extra_death <= 0.0:
        return 0.0

    extra = current_extra_death * (1.0 - resistance_score)
    if extra < 0.0:
        extra = 0.0
    return extra


@njit
def chooseElement(ListePopulation, total_pop):
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
    result = List()

    seed = np.zeros(7, dtype=np.int64)
    result.append(seed)
    result.pop()

    for i in range(len(ListePopulation)):
        entry = ListePopulation[i]
        if entry[POP_COUNT] > 0:
            new_entry = entry.copy()
            new_entry[POP_CELLS_TO_MUTATE] = 0
            new_entry[POP_FUSION_COUNT] = 0
            result.append(new_entry)

    return result


@njit
def fusionVect(G1, G2):
    hybrid1 = np.zeros(len(G1), dtype=np.bool_)
    hybrid2 = np.zeros(len(G1), dtype=np.bool_)

    for i in range(0, len(G1)):
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
    res = 0
    for i in range(0, len(ListePopulation)):
        genotype_idx = ListePopulation[i][POP_GENOTYPE]
        scorei = np.sum(all_genotypes[genotype_idx])
        res = scorei if res < scorei else res
    return res


def _write_csv_file(filepath: Path, header: list[str], data: list[list]) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def ModelRun(parameters: ModelParameters) -> ModelResult:
    Ngenes = parameters.number_of_genes
    NgenerationsMax = parameters.number_of_generations
    DT = parameters.dt
    DATA_RESOLUTION = parameters.data_resolution
    s = parameters.seed
    KC = parameters.carrying_capacity
    pm = parameters.mutation_rate_per_gene
    pf = parameters.fusion_rate
    growthRate = parameters.growth_rate
    deathRate = parameters.death_rate
    diversity = parameters.diversity if parameters.diversity is not None else 0
    initial_population_size = parameters.initial_population_size

    treatment_injection_every = parameters.treatment_injection_every
    treatment_initial_concentration = parameters.treatment_initial_concentration
    treatment_halflife = parameters.treatment_halflife
    treatment_concentration_to_extra_death = (
        parameters.treatment_concentration_to_extra_death
    )
    treatment_selection = parameters.treatment_selection
    treatment_resistivity = parameters.treatment_resistivity

    directory_path = parameters.save_path / f"g{growthRate}" / f"mu{pm}" / f"pf{pf}"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if s is not None:
        np.random.seed(int(s))

    lineage_data, metrics_data, all_genotypes = _ModelRun(
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
        treatment_injection_every=(
            treatment_injection_every if treatment_injection_every is not None else -1
        ),
        treatment_initial_concentration=treatment_initial_concentration,
        treatment_halflife=treatment_halflife,
        treatment_concentration_to_extra_death=treatment_concentration_to_extra_death,
        treatment_selection=treatment_selection,
        treatment_resistivity=treatment_resistivity,
    )

    # lineage stored internally as float arrays to allow fractional times.
    # convert back to mixed python rows for CSV writing.
    lineage_data_py = [
        [float(row[0]), int(row[1]), int(row[2]), int(row[3])] for row in lineage_data
    ]
    metrics_data_py = [row.tolist() for row in metrics_data]
    all_genotypes_py = [g.copy() for g in all_genotypes]

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
            MetricNames.drug_concentration,
            MetricNames.drug_extra_death_wt,
        ],
        metrics_data_py,
    )

    _write_csv_file(
        model_result.lineage_path,
        [
            MetricNames.time,
            MetricNames.cell_count,
            MetricNames.genotype_id,
            MetricNames.ancestor_id,
        ],
        lineage_data_py,
    )

    _write_csv_file(
        model_result.genotype_path,
        ["GenotypeId"] + [f"Locus{i}" for i in range(Ngenes)],
        [[idx] + genotype.tolist() for idx, genotype in enumerate(all_genotypes_py)],
    )

    return model_result


@njit
def same_genotype(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


@njit
def _ModelRun(
    Ngenes: int,
    NgenerationsMax: int,
    DT: float,
    DATA_RESOLUTION: int,
    KC: int,
    pm: float,
    pf: float,
    growthRate: float,
    deathRate: float,
    diversity: int = 0,
    initial_population_size: int = 1,
    treatment_injection_every: int = -1,
    treatment_initial_concentration: float = 0.25,
    treatment_halflife: float = 12.0,
    treatment_concentration_to_extra_death: float = 0.7 / 48.0,
    treatment_selection: float = 0.1,
    treatment_resistivity: float = 1.0,
):
    Genotype = np.zeros(Ngenes, dtype=np.bool_)

    Number = np.zeros(NgenerationsMax)
    Shanon = np.zeros(NgenerationsMax)
    Simpson = np.zeros(NgenerationsMax)
    Score = np.zeros(NgenerationsMax)
    TotalCells = np.zeros(NgenerationsMax)

    all_genotypes = List()
    genotype_resistance_scores = List()
    ListePop = List()
    ListeExtincted = List()

    # lineage data must allow fractional time, so store float64 rows
    lineage_data = List()
    metrics_data = List()
    genotypesCounts = 0

    seed_pop = np.zeros(7, dtype=np.int64)
    ListePop.append(seed_pop)
    ListePop.pop()

    seed_ext = np.zeros(7, dtype=np.int64)
    ListeExtincted.append(seed_ext)
    ListeExtincted.pop()

    seed_line = np.zeros(4, dtype=np.float64)
    lineage_data.append(seed_line)
    lineage_data.pop()

    seed_met = np.zeros(8, dtype=np.float64)
    metrics_data.append(seed_met)
    metrics_data.pop()

    seed_gen = np.zeros(Ngenes, dtype=np.bool_)
    all_genotypes.append(seed_gen)
    all_genotypes.pop()

    genotype_resistance_scores.append(0.0)
    genotype_resistance_scores.pop()

    resistivity = make_resistivity(Ngenes, treatment_selection, treatment_resistivity)
    decay_factor = compute_decay_factor(DT, treatment_halflife)
    current_concentration = 0.0

    all_genotypes.append(Genotype.copy())
    genotype_resistance_scores.append(
        genotype_resistance_score(Genotype, resistivity)
    )

    initial_entry = np.array(
        [initial_population_size, 0, 0, 0, genotypesCounts, 0, genotypesCounts], dtype=np.int64
    )
    ListePop.append(initial_entry)

    if diversity > 1:
        for i in range(1, diversity):
            genotypesCounts += 1
            mutant_genotype = TargetedMutation(Genotype, (i - 1) % Ngenes)
            all_genotypes.append(mutant_genotype)
            genotype_resistance_scores.append(
                genotype_resistance_score(mutant_genotype, resistivity)
            )
            diverse_entry = np.array(
                [1, len(all_genotypes) - 1, 0, 0, 0, 0, genotypesCounts], dtype=np.int64
            )
            ListePop.append(diverse_entry)

    for l in range(0, NgenerationsMax):
        current_time = l * DT
        TotalPopulation = countPopulation(ListePop)

        if is_injection_time(l, treatment_injection_every):
            current_concentration += treatment_initial_concentration

        current_extra_death_wt = (
            current_concentration * treatment_concentration_to_extra_death
        )

        if l % 200 == 0:
            print(
                "step =",
                l,
                "time_h =",
                current_time,
                "conc =",
                current_concentration,
                "extra_wt =",
                current_extra_death_wt,
                "TotalPopulation =",
                TotalPopulation,
            )

        for j in range(0, len(ListePop)):
            nombreRepresentants = ListePop[j][POP_COUNT]
            if nombreRepresentants > 0:
                newCells = np.random.poisson(growthRate * nombreRepresentants * DT)

                # mutation remains per birth event; do NOT rescale pm with dt
                p_mut = Ngenes * pm
                if p_mut < 0.0:
                    p_mut = 0.0
                if p_mut > 1.0:
                    p_mut = 1.0
                newM = np.random.binomial(newCells, p_mut)

                genotype_idx = ListePop[j][POP_GENOTYPE]

                base_term = deathRate * (TotalPopulation / KC)

                treat_term = 0.0
                if current_extra_death_wt > 0.0:
                    resistance_score = genotype_resistance_scores[genotype_idx]
                    treat_term = genotype_extra_death_from_score(
                        resistance_score,
                        current_extra_death_wt,
                    )

                hazard = (base_term + treat_term) * DT
                if hazard < 0.0:
                    hazard = 0.0

                newD = np.random.poisson(hazard * nombreRepresentants)

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
                    extinct_entry = np.array(
                        [
                            1,
                            ListePop[j][POP_GENOTYPE],
                            ListePop[j][POP_CELLS_TO_MUTATE],
                            int(current_time),
                            ListePop[j][POP_ANCESTOR_ID],
                            ListePop[j][POP_FUSION_COUNT],
                            ListePop[j][POP_GENOTYPE_ID],
                        ],
                        dtype=np.int64,
                    )
                    ListeExtincted.append(extinct_entry)

        j = 0
        Ncurrent = len(ListePop)
        while j < Ncurrent:
            for _ in range(0, ListePop[j][POP_CELLS_TO_MUTATE]):
                parent_genotype = all_genotypes[ListePop[j][POP_GENOTYPE]]
                GenotypeTemporaire = Mutation(parent_genotype)
                exist = 0
                count = 0
                while count < len(ListePop) and exist == 0:
                    if same_genotype(
                        all_genotypes[ListePop[count][POP_GENOTYPE]], GenotypeTemporaire
                    ):
                        ListePop[count][POP_COUNT] = ListePop[count][POP_COUNT] + 1
                        exist = 1
                    count = count + 1
                if exist == 0:
                    genotypesCounts = genotypesCounts + 1
                    all_genotypes.append(GenotypeTemporaire)
                    genotype_resistance_scores.append(
                        genotype_resistance_score(GenotypeTemporaire, resistivity)
                    )
                    new_genotype_entry = np.array(
                        [
                            1,
                            len(all_genotypes) - 1,
                            0,
                            int(current_time),
                            ListePop[j][POP_GENOTYPE_ID],
                            0,
                            genotypesCounts,
                        ],
                        dtype=np.int64,
                    )
                    ListePop.append(new_genotype_entry)
            ListePop[j][POP_CELLS_TO_MUTATE] = 0
            j = j + 1

        TotalPopulation = countPopulation(ListePop)

        # fusion rate is per unit time, so it should be rescaled when dt changes
        newH = np.random.poisson(pf * TotalPopulation * DT)
        newH = int(np.minimum(newH, int(TotalPopulation / 2)))

        for _ in range(0, newH):
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

                    if ListePop[count][POP_COUNT] == 0:
                        extinct_entry = np.array(
                            [
                                1,
                                ListePop[count][POP_GENOTYPE],
                                ListePop[count][POP_CELLS_TO_MUTATE],
                                int(current_time),
                                ListePop[count][POP_ANCESTOR_ID],
                                ListePop[count][POP_FUSION_COUNT],
                                ListePop[count][POP_GENOTYPE_ID],
                            ],
                            dtype=np.int64,
                        )
                        ListeExtincted.append(extinct_entry)
                count = count + 1

        Ncurrent = len(ListePop)
        TotalPopulation = countPopulation(ListePop)

        if TotalPopulation > 0:
            for j in range(0, Ncurrent):
                for _ in range(0, ListePop[j][POP_FUSION_COUNT]):
                    Genotype1 = all_genotypes[ListePop[j][POP_GENOTYPE]]

                    neighbor = chooseElement(ListePop, TotalPopulation)

                    ListePop[j][POP_COUNT] = ListePop[j][POP_COUNT] - 1
                    ListePop[neighbor][POP_COUNT] = ListePop[neighbor][POP_COUNT] - 1
                    TotalPopulation = TotalPopulation - 2

                    if ListePop[j][POP_COUNT] == 0:
                        extinct_entry = np.array(
                            [
                                1,
                                ListePop[j][POP_GENOTYPE],
                                ListePop[j][POP_CELLS_TO_MUTATE],
                                int(current_time),
                                ListePop[j][POP_ANCESTOR_ID],
                                ListePop[j][POP_FUSION_COUNT],
                                ListePop[j][POP_GENOTYPE_ID],
                            ],
                            dtype=np.int64,
                        )
                        ListeExtincted.append(extinct_entry)

                    if ListePop[neighbor][POP_COUNT] == 0:
                        extinct_entry = np.array(
                            [
                                1,
                                ListePop[neighbor][POP_GENOTYPE],
                                ListePop[neighbor][POP_CELLS_TO_MUTATE],
                                int(current_time),
                                ListePop[neighbor][POP_ANCESTOR_ID],
                                ListePop[neighbor][POP_FUSION_COUNT],
                                ListePop[neighbor][POP_GENOTYPE_ID],
                            ],
                            dtype=np.int64,
                        )
                        ListeExtincted.append(extinct_entry)

                    Genotype2 = all_genotypes[ListePop[neighbor][POP_GENOTYPE]]

                    hybrid1, hybrid2 = fusionVect(Genotype1, Genotype2)

                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if same_genotype(
                            all_genotypes[ListePop[count][POP_GENOTYPE]], hybrid1
                        ):
                            ListePop[count][POP_COUNT] += 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1
                        all_genotypes.append(hybrid1)
                        genotype_resistance_scores.append(
                            genotype_resistance_score(hybrid1, resistivity)
                        )
                        hybrid_entry = np.array(
                            [
                                1,
                                len(all_genotypes) - 1,
                                0,
                                int(current_time),
                                ListePop[j][POP_GENOTYPE_ID],
                                0,
                                genotypesCounts,
                            ],
                            dtype=np.int64,
                        )
                        ListePop.append(hybrid_entry)

                    exist = 0
                    count = 0
                    while count < len(ListePop) and exist == 0:
                        if same_genotype(
                            all_genotypes[ListePop[count][POP_GENOTYPE]], hybrid2
                        ):
                            ListePop[count][POP_COUNT] += 1
                            exist = 1
                        count = count + 1
                    if exist == 0:
                        genotypesCounts = genotypesCounts + 1
                        all_genotypes.append(hybrid2)
                        genotype_resistance_scores.append(
                            genotype_resistance_score(hybrid2, resistivity)
                        )
                        hybrid_entry = np.array(
                            [
                                1,
                                len(all_genotypes) - 1,
                                0,
                                int(current_time),
                                ListePop[j][POP_GENOTYPE_ID],
                                0,
                                genotypesCounts,
                            ],
                            dtype=np.int64,
                        )
                        ListePop.append(hybrid_entry)

                    TotalPopulation = TotalPopulation + 2

                ListePop[j][POP_FUSION_COUNT] = 0

        else:
            for j in range(0, Ncurrent):
                for _ in range(0, ListePop[j][POP_FUSION_COUNT]):
                    ListePop[j][POP_COUNT] = ListePop[j][POP_COUNT] + 1

        ListePop = cleanData(ListePop)

        if l % DATA_RESOLUTION == 0 or l == NgenerationsMax - 1:
            Number[l] = countSpecies(ListePop)
            TotalCells[l] = countPopulation(ListePop)
            Shanon[l] = ComputeShanon(ListePop)
            Simpson[l] = ComputeIndex(ListePop, 2)
            Score[l] = MaxScore(ListePop, all_genotypes)

            for i in range(0, len(ListeExtincted)):
                row = np.array(
                    [
                        current_time,
                        1.0,
                        float(ListeExtincted[i][POP_GENOTYPE_ID]),
                        float(ListeExtincted[i][POP_ANCESTOR_ID]),
                    ],
                    dtype=np.float64,
                )
                lineage_data.append(row)

            for i in range(0, len(ListePop)):
                row = np.array(
                    [
                        current_time,
                        float(ListePop[i][POP_COUNT]),
                        float(ListePop[i][POP_GENOTYPE_ID]),
                        float(ListePop[i][POP_ANCESTOR_ID]),
                    ],
                    dtype=np.float64,
                )
                lineage_data.append(row)

            row = np.array(
                [
                    float(current_time),
                    float(TotalCells[l]),
                    float(Number[l]),
                    float(Shanon[l]),
                    float(Simpson[l]),
                    float(Score[l]),
                    float(current_concentration),
                    float(current_extra_death_wt),
                ],
                dtype=np.float64,
            )
            metrics_data.append(row)

            while len(ListeExtincted) > 0:
                ListeExtincted.pop()

        current_concentration *= decay_factor

    return lineage_data, metrics_data, all_genotypes


if __name__ == "__main__":
    start = time.time()

    # With dt = 0.25 hours (15 min), rates that were previously specified per 12h
    # should usually be divided by 48, EXCEPT mutation_rate_per_gene which is per birth.
    params = ModelParameters(
        seed=int(sys.argv[1]),
        number_of_genes=int(sys.argv[2]),
        carrying_capacity=int(sys.argv[3]),
        number_of_generations=int(sys.argv[4]),
        mutation_rate_per_gene=float(sys.argv[5]),  # per birth event; not rescaled by dt
        fusion_rate=float(sys.argv[6]),  # per hour-scale unit used in the simulator
        growth_rate=float(sys.argv[7]),
        death_rate=float(sys.argv[8]),
        save_path=Path("Results") / f"CC{int(sys.argv[3])}" / "Neutral/LogisticFusion",
        diversity=int(sys.argv[9]),
        dt=0.25,
        data_resolution=4,
        treatment_injection_every=21 * 24 * 4,
        treatment_initial_concentration=0.25,
        treatment_halflife=12.0,
        treatment_concentration_to_extra_death=0.7 / 48.0,
        treatment_selection=0.1,
        treatment_resistivity=1.0,
    )

    ModelRun(parameters=params)

    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))