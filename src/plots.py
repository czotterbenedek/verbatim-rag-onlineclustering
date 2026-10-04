from pathlib import Path


def _save_bar_plot(path, title, ylabel, labels, values, value_format="{:.2f}"):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    positions = range(len(labels))
    bars = axis.bar(positions, values, color="#3b82f6", width=0.65)
    axis.set(title=title, xlabel="System", ylabel=ylabel, xticks=list(positions), xticklabels=labels)
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axis.annotate(value_format.format(value),
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 4), textcoords="offset points", ha="center")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_plots(output_directory, run_records):
    """Create the required quality/efficiency plots from saved run summaries."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = list(run_records)
    fractions = [run_records[label]["metrics"]["candidate_fraction"] for label in labels]
    hit_at_10 = [run_records[label]["metrics"]["gold_hit_at_10"] for label in labels]
    retrieval_latency = [run_records[label]["metrics"]["retrieval_latency_ms"] for label in labels]
    speedup = [run_records[label]["metrics"].get("retrieval_speedup_vs_baseline") or 1.0 for label in labels]
    routing_recall = [run_records[label]["metrics"].get("routing_hit") or 0.0 for label in labels]
    overlap = [run_records[label]["metrics"].get("baseline_overlap_at_k") or 1.0 for label in labels]

    _save_bar_plot(output_directory / "quality_vs_candidate_fraction.png",
                   "Retrieval quality by candidate fraction", "Gold Hit@10",
                   labels, hit_at_10, "{:.1%}")
    _save_bar_plot(output_directory / "latency_vs_candidate_fraction.png",
                   "Retrieval latency by candidate fraction", "Retrieval latency (ms)",
                   labels, retrieval_latency, "{:.2f}")
    _save_bar_plot(output_directory / "routing_recall_vs_clusters_searched.png",
                   "Cluster routing recall by system", "Routing recall",
                   labels, routing_recall, "{:.1%}")
    _save_bar_plot(output_directory / "baseline_overlap_vs_candidate_fraction.png",
                   "Baseline preservation by candidate fraction", "Overlap@K",
                   labels, overlap, "{:.1%}")
    _save_bar_plot(output_directory / "candidate_reduction_by_system.png",
                   "Search-space reduction by system", "Candidate reduction",
                   labels, [1.0 - value for value in fractions], "{:.1%}")
    _save_bar_plot(output_directory / "cluster_count_by_system.png",
                   "Configured cluster count by system", "Number of clusters",
                   labels, [run_records[label]["metrics"].get("num_clusters", 0) for label in labels], "{:.0f}")
    _save_bar_plot(output_directory / "retrieval_speedup_by_system.png",
                   "Verbatim-RAG retrieval speed-up by system", "Speed-up vs baseline (x)",
                   labels, speedup, "{:.2f}x")


def _save_growth_line_plot(path, title, ylabel, x_values, series, value_format="{:.2f}"):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for label, values in series.items():
        axis.plot(x_values, values, marker="o", linewidth=2, label=label)
        for x_value, value in zip(x_values, values):
            axis.annotate(value_format.format(value), (x_value, value),
                          xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axis.set(title=title, xlabel="Visible chunks", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_growth_plots(output_directory, stage_records):
    """Create longitudinal plots from staged growth metrics."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if not stage_records:
        return
    x_values = [record["num_chunks"] for record in stage_records]
    systems = sorted({
        system for record in stage_records for system in record.get("metrics", {})
    })

    def metric_series(metric, default=0.0):
        return {
            system: [
                record.get("metrics", {}).get(system, {}).get(metric)
                if record.get("metrics", {}).get(system, {}).get(metric) is not None
                else default
                for record in stage_records
            ]
            for system in systems
        }

    _save_growth_line_plot(
        output_directory / "quality_vs_corpus_size.png",
        "Retrieval quality as the corpus grows", "Gold Hit@10", x_values,
        metric_series("gold_hit_at_10"), "{:.1%}"
    )
    _save_growth_line_plot(
        output_directory / "candidate_fraction_vs_corpus_size.png",
        "Candidate fraction as the corpus grows", "Candidate fraction", x_values,
        metric_series("candidate_fraction"), "{:.1%}"
    )
    _save_growth_line_plot(
        output_directory / "retrieval_latency_vs_corpus_size.png",
        "Retrieval latency as the corpus grows", "Retrieval latency (ms)", x_values,
        metric_series("retrieval_latency_ms"), "{:.2f}"
    )
    _save_growth_line_plot(
        output_directory / "routing_recall_vs_corpus_size.png",
        "Routing recall as the corpus grows", "Routing recall", x_values,
        metric_series("routing_hit"), "{:.1%}"
    )
    _save_growth_line_plot(
        output_directory / "speedup_vs_corpus_size.png",
        "Retrieval speed-up versus full-search baseline", "Speed-up (x)", x_values,
        metric_series("retrieval_speedup_vs_baseline", 1.0), "{:.2f}x"
    )

    update_x = [record["update"]["num_chunks_after"] for record in stage_records]
    update_series = {
        "Online update": [record["update"]["total_latency_ms"] for record in stage_records],
        "Offline refit": [record["update"].get("offline_fit_latency_ms") or 0.0
                           for record in stage_records],
    }
    _save_growth_line_plot(
        output_directory / "update_cost_vs_corpus_size.png",
        "Incremental update versus offline refit cost", "Indexing time (ms)",
        update_x, update_series, "{:.1f}"
    )
    _save_growth_line_plot(
        output_directory / "online_throughput_vs_corpus_size.png",
        "Online update throughput", "New chunks per second", update_x,
        {"Online update": [record["update"]["throughput_chunks_per_second"]
                            for record in stage_records]}, "{:.0f}"
    )
    _save_growth_line_plot(
        output_directory / "centroid_drift_vs_corpus_size.png",
        "Centroid drift after online updates", "Mean centroid drift", update_x,
        {"Online K-Means": [record["update"]["centroid_drift_mean"]
                            for record in stage_records]}, "{:.3f}"
    )
    _save_growth_line_plot(
        output_directory / "query_coverage_vs_corpus_size.png",
        "Evaluated query coverage", "Available questions", update_x,
        {"Questions": [record["update"]["num_questions"] for record in stage_records]}, "{:.0f}"
    )