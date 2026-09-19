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