"""File to coordinate the running of experiments for the non-spatial model."""

import time
from non_spatial.NonSpatialFusion import ModelRun
from non_spatial.parametrization import ModelParameters
from output import OUTPUT_PATH

# Time step = 15 minutes = 0.25 hours
# Anything that was "per 12 hours" should be divided by 48 to preserve biology,
# except mutation_rate_per_gene, which is per birth event in this model.

# Parameters from monte_carlo_param_sweep.py with initial_population_size=10000 and treatment_initial_concentration=0
baseline_params = ModelParameters(
    number_of_genes=100,
    carrying_capacity=100000,
    number_of_generations=24 * 4 * 180,  # 180 days, 15-minute steps
    mutation_rate_per_gene=1e-4,  # per birth event -> unchanged
    fusion_rate=1.4e-3,  # rescaled from per-12h to per-15min
    growth_rate=0.12 / 48.0,  # rescaled from per-12h to per-15min
    death_rate=0.04 / 48.0,  # rescaled from per-12h to per-15min
    save_path=OUTPUT_PATH,
    dt=0.25,  # 15 minutes = 0.25 hours
    data_resolution=24 * 4,  # store every day
    diversity=1,
    initial_population_size=10000,  # 1e4
    seed=0,
    treatment_injection_every=21 * 24 * 4,  # every 3 weeks
    treatment_initial_concentration=0,  # NO TREATMENT
    treatment_halflife=12.0,  # 12 hours
    treatment_concentration_to_extra_death=0.7,
    treatment_selection=0.1,
    treatment_resistivity=1.0,
    treatment_epistasis=1.0,
)

print("=" * 70)
print("TIMING 10 RUNS FOR ROBUST ESTIMATION")
print("=" * 70)
print(f"Parameters: 1e4 initial pop, no treatment, 180 days")
print(f"Running 10 seeds to estimate variance and mean runtime")
print()

run_times = []
for i in range(10):
    print(f"\nRun {i + 1}/10...", end=" ", flush=True)
    start_time = time.time()
    result = ModelRun(parameters=baseline_params)
    elapsed_time = time.time() - start_time
    run_times.append(elapsed_time)
    print(f"{elapsed_time:.2f}s")

import numpy as np

mean_time = np.mean(run_times)
std_time = np.std(run_times)
min_time = np.min(run_times)
max_time = np.max(run_times)

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(f"Mean time per seed:       {mean_time:.2f} seconds")
print(f"Std deviation:            {std_time:.2f} seconds")
print(f"Min time:                 {min_time:.2f} seconds")
print(f"Max time:                 {max_time:.2f} seconds")
print(f"\nEstimated time for 10,000 seeds:")
print(
    f"  Mean estimate:          {mean_time * 10000 / 3600:.2f} hours ({mean_time * 10000 / 86400:.2f} days)"
)
print(
    f"  Worst case (+1σ):       {(mean_time + std_time) * 10000 / 3600:.2f} hours ({(mean_time + std_time) * 10000 / 86400:.2f} days)"
)
print(
    f"  Best case (-1σ):        {(mean_time - std_time) * 10000 / 3600:.2f} hours ({(mean_time - std_time) * 10000 / 86400:.2f} days)"
)
print("=" * 70)
