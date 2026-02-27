"""analysis notebook. To run, run `marimo edit non_spatial_no_chemotherapy/run_analysis.py` from project root."""

import marimo


__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import sys

    # Make parent folder visible so we can import output.py
    sys.path.append(".")

    # Import local modules (same folder)
    from NonSpatialFusion import ModelRun
    from parametrization import ModelParameters, MetricNames
    from output import OUTPUT_PATH

    return MetricNames, ModelParameters, ModelRun, OUTPUT_PATH, pl


@app.cell
def _(ModelParameters, ModelRun, OUTPUT_PATH):
    def run_one(seed: int):
        params = ModelParameters(
            number_of_genes=100,
            carrying_capacity=1000,
            dt=0.08,
            number_of_generations=5000,
            mutation_rate_per_gene=1e-4,
            fusion_rate=1e-4,
            growth_rate=0.12,
            death_rate=0.08,
            save_path=OUTPUT_PATH,
            seed=seed,
            treatment_every=500,
            treatment_duration=500,
            treatment_base_extra_death=0.1,
            treatment_selection=0.1,
        )
        return ModelRun(params)

    return (run_one,)


@app.cell
def _(run_one):
    seeds = [0]
    results = []
    for _s in seeds:
        print("Running seed", _s)
        _r = run_one(_s)
        results.append(_r)
        print("  metrics:", _r.metrics_path)
        print("  lineage:", _r.lineage_path)
        print("  genotypes:", _r.genotype_path)
    return results, seeds


@app.cell
def _(MetricNames, pl, results, seeds):
    import matplotlib.pyplot as plt

    TOP_N = 10**9
    for _s, _r in zip(seeds, results):  # effectively "all strands"
        lineage = pl.read_csv(_r.lineage_path)
        grouped = (
            lineage.group_by([MetricNames.time, MetricNames.genotype_id])
            .agg(pl.col(MetricNames.cell_count).sum())
            .sort([MetricNames.time, MetricNames.genotype_id])
        )
        grouped = (
            grouped.with_columns(
                pl.col(MetricNames.cell_count)
                .rank(descending=True)
                .over(MetricNames.time)
                .alias("rank_at_time")
            )
            .filter(pl.col("rank_at_time") <= TOP_N)
            .drop("rank_at_time")
        )
        d = grouped.to_dict(as_series=False)
        times_list = d[
            MetricNames.time
        ]  # same grouping logic as your plot_results.plot_lineage_stack
        genotypes_list = d[MetricNames.genotype_id]
        counts_list = d[MetricNames.cell_count]
        lookup = dict(zip(zip(times_list, genotypes_list), counts_list))
        _times = sorted(set(times_list))
        genotypes = sorted(set(genotypes_list))
        stackplot_data = []
        for g in genotypes:
            stackplot_data.append([lookup.get((t, g), 0) for t in _times])
        plt.figure(figsize=(14, 8))
        plt.stackplot(_times, *stackplot_data, alpha=0.8)
        plt.title(f"Lineage composition over time (seed={_s})")
        plt.xlabel("Time (days)")
        plt.ylabel("Cell count")
        plt.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.show()
    return (plt,)


@app.cell
def _(MetricNames, pl, plt, results, seeds):
    for _s, _r in zip(seeds, results):
        _metrics = pl.read_csv(_r.metrics_path)
        _x = _metrics[MetricNames.time].to_numpy()
        y = _metrics[MetricNames.total_cells].to_numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(_x, y, marker="o", linewidth=2)
        plt.title(f"Total population over time (seed={_s})")
        plt.xlabel("Time")
        plt.ylabel("Total cells")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.show()
    return


@app.cell
def _(MetricNames, pl, plt, results, seeds):
    for _s, _r in zip(seeds, results):
        _metrics = pl.read_csv(_r.metrics_path)
        _x = _metrics[MetricNames.time].to_numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(
            _x, _metrics[MetricNames.num_genotypes].to_numpy(), marker="o", linewidth=2
        )
        plt.title(f"Number of genotypes over time (seed={_s})")
        plt.xlabel("Time")
        plt.ylabel("Num genotypes")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.plot(
            _x,
            _metrics[MetricNames.shannon_index].to_numpy(),
            marker="o",
            linewidth=2,
            label=MetricNames.shannon_index,
        )
        plt.plot(
            _x,
            _metrics[MetricNames.simpson_index].to_numpy(),
            marker="o",
            linewidth=2,
            label=MetricNames.simpson_index,
        )
        plt.title(f"Diversity indices over time (seed={_s})")
        plt.xlabel("Time")
        plt.ylabel("Index value")
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.plot(
            _x, _metrics[MetricNames.max_mutations].to_numpy(), marker="o", linewidth=2
        )
        plt.title(f"Max mutations over time (seed={_s})")
        plt.xlabel("Time")
        plt.ylabel("Max mutations")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.show()
    return


@app.cell
def _(pl, plt, results, seeds):
    import numpy as np

    GENO_N = 200
    for _s, _r in zip(seeds, results):
        genos = pl.read_csv(_r.genotype_path)
        data = genos.filter(pl.col("GenotypeId") < GENO_N)
        locus_columns = [col for col in data.columns if col.startswith("Locus")]
        genotype_ids = data["GenotypeId"].to_list()
        locus_data = data.select(locus_columns).cast(pl.Int32).to_numpy()
        num_genotypes = len(genotype_ids)
        num_loci = len(locus_columns)
        figsize = (max(12, num_loci / 3), max(6, num_genotypes / 2))
        plt.figure(figsize=figsize)
        plt.imshow(
            locus_data,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        plt.gca().set_xticks(np.arange(-0.5, num_loci, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, num_genotypes, 1), minor=True)
        plt.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.6)
        plt.tick_params(which="minor", size=0)
        plt.xlabel("Locus")
        locus_indices = np.arange(0, num_loci, max(1, num_loci // 10))
        plt.xticks(locus_indices, [int(i) for i in locus_indices], fontsize=9)
        plt.ylabel("Genotype ID")
        if num_genotypes <= 50:
            plt.yticks(np.arange(num_genotypes), genotype_ids, fontsize=8)
        else:
            label_interval = max(1, num_genotypes // 20)
            label_indices = np.arange(0, num_genotypes, label_interval)
            plt.yticks(
                label_indices, [genotype_ids[int(i)] for i in label_indices], fontsize=8
            )
        plt.title(f"Genotype mutation patterns (seed={_s}) ù first {GENO_N}")
        plt.show()
    return


@app.cell
def _(MetricNames, pl, results, seeds):
    def is_treatment_time(t: int, every: int, duration: int) -> bool:
        if every is None or duration <= 0:
            return False
        cycle = every + duration
        return t % cycle >= every

    EVERY = 20
    DUR = 20
    for _s, _r in zip(seeds, results):
        _metrics = pl.read_csv(_r.metrics_path)
        _times = _metrics[MetricNames.time].to_list()
        total = _metrics[MetricNames.total_cells].to_list()
        on = [
            total[i]
            for i, t in enumerate(_times)
            if is_treatment_time(int(t), EVERY, DUR)
        ]
        off = [
            total[i]
            for i, t in enumerate(_times)
            if not is_treatment_time(int(t), EVERY, DUR)
        ]
        print(
            f"seed={_s}: mean TotalCells on-tx={sum(on) / len(on):.2f} off-tx={sum(off) / len(off):.2f} (n_on={len(on)}, n_off={len(off)})"
        )
    return


if __name__ == "__main__":
    app.run()
