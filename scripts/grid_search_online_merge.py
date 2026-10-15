#!/usr/bin/env python
"""Grid-search Online K-Means creation and merge thresholds."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster_index import ClusteredRetrievalIndex
from src.config import config_path, load_config
from src.data_pipeline import read_jsonl
from src.evaluation import evaluate, summarize
from scripts.run_online_growth import stage_endpoints


def parse_thresholds(value):
    thresholds = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        threshold = float(item)
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError(
                "Cosine thresholds must be between 0 and 1."
            )
        thresholds.append(threshold)
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    return thresholds


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


OBJECTIVE_DIRECTIONS = {
    "gold_hit_at_10": "maximize",
    "routing_hit": "maximize",
    "candidate_fraction": "minimize",
    "latency_ms": "minimize",
    "retrieval_latency_ms": "minimize",
    "update_total_latency_ms": "minimize",
    "num_clusters": "minimize",
}


def select_best_params(rows, objective):
    """Select the best threshold pair from the final growth stage."""
    final_stage = max(row["stage"] for row in rows)
    candidates = [
        row for row in rows
        if row["stage"] == final_stage and row[objective] is not None
    ]
    if not candidates:
        raise ValueError(
            f"No rows with a value for objective {objective!r} at the final stage."
        )

    maximize = OBJECTIVE_DIRECTIONS[objective] == "maximize"

    def sort_key(row):
        objective_value = row[objective]
        primary = -objective_value if maximize else objective_value
        return (
            primary,
            -row["gold_hit_at_10"] if row["gold_hit_at_10"] is not None else float("inf"),
            row["latency_ms"] if row["latency_ms"] is not None else float("inf"),
            row["update_total_latency_ms"]
            if row["update_total_latency_ms"] is not None
            else float("inf"),
            row["new_cluster_threshold"],
            row["merge_threshold"],
        )

    best = min(candidates, key=sort_key)
    return {
        "objective": objective,
        "direction": OBJECTIVE_DIRECTIONS[objective],
        "stage": best["stage"],
        "num_chunks": best["num_chunks"],
        "new_cluster_threshold": best["new_cluster_threshold"],
        "merge_threshold": best["merge_threshold"],
        "metrics": {
            key: value
            for key, value in best.items()
            if key not in {"new_cluster_threshold", "merge_threshold"}
        },
    }


def create_grid_plots(output_directory, rows):
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_pairs = sorted({
        (row["new_cluster_threshold"], row["merge_threshold"])
        for row in rows
    })
    stages = sorted({row["num_chunks"] for row in rows})
    colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_pairs)))

    def pair_label(pair):
        return f"new={pair[0]:.4f}, merge={pair[1]:.4f}"

    def plot_stage_metric(filename, title, ylabel, field, value_format=None):
        figure, axis = plt.subplots(figsize=(9, 5))
        for color, parameter_pair in zip(colors, parameter_pairs):
            new_threshold, merge_threshold = parameter_pair
            values = [
                next(row[field] for row in rows
                     if row["new_cluster_threshold"] == new_threshold
                     and row["merge_threshold"] == merge_threshold
                     and row["num_chunks"] == stage)
                for stage in stages
            ]
            axis.plot(stages, values, marker="o", linewidth=2, color=color,
                      label=pair_label(parameter_pair))
            if value_format:
                for stage, value in zip(stages, values):
                    axis.annotate(value_format.format(value), (stage, value),
                                  xytext=(0, 6), textcoords="offset points",
                                  ha="center", fontsize=7)
        axis.set(title=title, xlabel="Visible chunks", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(title="threshold pair", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=160)
        plt.close(figure)

    plot_stage_metric(
        "hit_at_10_vs_corpus_size.png",
        "Hit@10 by threshold pair as the corpus grows",
        "Gold Hit@10", "gold_hit_at_10", "{:.1%}",
    )
    plot_stage_metric(
        "routing_recall_vs_corpus_size.png",
        "Routing recall by threshold pair as the corpus grows",
        "Routing recall", "routing_hit", "{:.1%}",
    )
    plot_stage_metric(
        "retrieval_latency_vs_corpus_size.png",
        "Retrieval latency by threshold pair as the corpus grows",
        "Retrieval latency (ms)", "retrieval_latency_ms", "{:.1f}",
    )
    plot_stage_metric(
        "update_time_vs_corpus_size.png",
        "Online update time by threshold pair as the corpus grows",
        "Update time (ms)", "update_total_latency_ms", "{:.1f}",
    )
    plot_stage_metric(
        "cluster_count_vs_corpus_size.png",
        "Cluster count after creation/merging as the corpus grows",
        "Number of clusters", "num_clusters", "{:.0f}",
    )

    final_rows = [
        next(row for row in rows
             if (row["new_cluster_threshold"], row["merge_threshold"]) == pair
             and row["num_chunks"] == stages[-1])
        for pair in parameter_pairs
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    final_metrics = (
        ("gold_hit_at_10", "Hit@10", "{:.1%}"),
        ("retrieval_latency_ms", "Retrieval latency (ms)", "{:.1f}"),
        ("update_total_latency_ms", "Update time (ms)", "{:.1f}"),
        ("num_clusters", "Number of clusters", "{:.0f}"),
    )
    for axis, (field, title, value_format) in zip(axes.flat, final_metrics):
        values = [row[field] for row in final_rows]
        positions = np.arange(len(parameter_pairs))
        axis.plot(positions, values, marker="o", linewidth=2)
        axis.set(title=f"Final stage: {title}", xlabel="threshold pair",
                 xticks=positions,
                 xticklabels=[pair_label(pair) for pair in parameter_pairs])
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        for position, value in zip(positions, values):
            axis.annotate(value_format.format(value), (position, value),
                          xytext=(0, 6), textcoords="offset points",
                          ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_directory / "final_metrics_vs_threshold_pair.png", dpi=160
    )
    plt.close(figure)


def load_growth_data(config):
    data, embedding = config["data"], config["embedding"]
    chunks = read_jsonl(config_path(config, data["chunks_path"]))
    questions = read_jsonl(config_path(config, data["evaluation_path"]))
    vectors = np.load(config_path(config, embedding["path"]))
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding/chunk mismatch: {vectors.shape[0]} embeddings for "
            f"{len(chunks)} chunks. Run scripts/create_embeddings.py again."
        )

    query_path = config_path(config, embedding["query_path"])
    query_vectors = np.load(query_path)
    query_names = json.loads(query_path.with_suffix(".json").read_text(encoding="utf-8"))
    query_map = dict(zip(query_names, query_vectors))
    return chunks, questions, vectors, query_map


def run_threshold(
    config, chunks, questions, vectors, query_map,
    new_cluster_threshold, merge_threshold,
):
    growth = config.get("online_growth", {})
    clustering = config["clustering"]
    order_name = growth.get("order", "fixed")
    if order_name == "shuffled":
        rng = np.random.RandomState(growth.get("stream_seed", 42))
        order = rng.permutation(len(chunks))
    elif order_name == "fixed":
        order = np.arange(len(chunks))
    else:
        raise ValueError("online_growth.order must be 'fixed' or 'shuffled'.")

    ordered_chunks = [chunks[int(index)] for index in order]
    ordered_vectors = vectors[order]
    endpoints = stage_endpoints(
        len(ordered_chunks), growth.get("stages", [1.0])
    )
    if endpoints[0] < clustering["n_clusters"]:
        raise ValueError(
            "The first growth stage must contain at least n_clusters embeddings."
        )

    index = None
    previous_end = 0
    rows = []
    for stage_number, stage_end in enumerate(endpoints):
        stage_chunks = ordered_chunks[:stage_end]
        stage_vectors = ordered_vectors[:stage_end]
        new_vectors = ordered_vectors[previous_end:stage_end]
        visible_papers = {str(chunk["paper_id"]) for chunk in stage_chunks}
        stage_questions = [
            row for row in questions
            if str(row.get("gold_paper")) in visible_papers
        ]

        update_started = time.perf_counter()
        if index is None:
            index = ClusteredRetrievalIndex(
                new_vectors,
                clustering["n_clusters"],
                clustering["metric"],
                clustering["random_state"],
                growth.get("update_batch_size", clustering["batch_size"]),
                "online_kmeans",
                clustering.get("max_clusters"),
                new_cluster_threshold,
                merge_threshold,
                clustering.get("split_conductance_threshold")
                if clustering.get("splitting", False)
                else None,
            )
            update_latency_ms = (time.perf_counter() - update_started) * 1000
            update = {
                "update_latency_ms": update_latency_ms,
                "assignment_latency_ms": 0.0,
                "total_latency_ms": update_latency_ms,
                "num_clusters": int(len(index.kmeans.centroids)),
                "total_seen": int(index.kmeans.total_seen),
            }
        else:
            update = index.append_embeddings(
                new_vectors, growth.get("update_batch_size", clustering["batch_size"])
            )

        predictions = evaluate(
            stage_questions,
            stage_chunks,
            stage_vectors,
            query_map,
            config["retrieval"]["top_k"],
            index=index,
            top_clusters=clustering["top_clusters"],
            system_name="online_kmeans",
        )
        metrics = summarize(predictions)
        rows.append({
            "new_cluster_threshold": new_cluster_threshold,
            "merge_threshold": merge_threshold,
            "stage": stage_number,
            "num_chunks": stage_end,
            "num_questions": len(stage_questions),
            "num_clusters": update["num_clusters"],
            "gold_hit_at_1": metrics.get("gold_hit_at_1"),
            "gold_hit_at_5": metrics.get("gold_hit_at_5"),
            "gold_hit_at_10": metrics.get("gold_hit_at_10"),
            "routing_hit": metrics.get("routing_hit"),
            "candidate_fraction": metrics.get("candidate_fraction"),
            "latency_ms": metrics.get("latency_ms"),
            "retrieval_latency_ms": metrics.get("retrieval_latency_ms"),
            "update_latency_ms": update["update_latency_ms"],
            "assignment_latency_ms": update["assignment_latency_ms"],
            "update_total_latency_ms": update["total_latency_ms"],
        })
        previous_end = stage_end

    return rows


def print_table(rows):
    headers = (
        "new_threshold", "merge_threshold", "stage", "questions", "clusters", "Hit@10",
        "delta_Hit@10", "routing", "candidate%", "latency_ms",
        "delta_latency_ms", "retrieval_ms", "update_ms", "delta_update_ms",
    )
    print("\t".join(headers))
    for row in rows:
        values = (
            f"{row['new_cluster_threshold']:.4f}",
            f"{row['merge_threshold']:.4f}",
            str(row["stage"]),
            str(row["num_questions"]),
            str(row["num_clusters"]),
            f"{row['gold_hit_at_10']:.4f}",
            f"{row['delta_gold_hit_at_10_vs_first_threshold']:+.4f}",
            f"{row['routing_hit']:.4f}" if row["routing_hit"] is not None else "n/a",
            f"{100 * row['candidate_fraction']:.2f}",
            f"{row['latency_ms']:.2f}",
            f"{row['delta_latency_ms_vs_first_threshold']:+.2f}",
            f"{row['retrieval_latency_ms']:.2f}",
            f"{row['update_total_latency_ms']:.2f}",
            f"{row['delta_update_total_latency_ms_vs_first_threshold']:+.2f}",
        )
        print("\t".join(values))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search Online K-Means new-cluster and merge thresholds "
            "in online growth."
        )
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--merge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.01,0.02,0.05,0.10"),
        help="Comma-separated cosine-distance thresholds in [0, 1].",
    )
    parser.add_argument(
        "--new-cluster-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.05,0.10,0.20"),
        help="Comma-separated cosine-distance thresholds in [0, 1].",
    )
    parser.add_argument(
        "--objective",
        choices=tuple(OBJECTIVE_DIRECTIONS),
        default="gold_hit_at_10",
        help=(
            "Metric used to select the best threshold pair at the final growth "
            "stage. Quality metrics are maximized and cost metrics minimized."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output directory. Defaults to experiments/<name>_<timestamp>.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config["clustering"].get("metric") != "cosine":
        raise ValueError(
            "This script expects clustering.metric: cosine because thresholds "
            "are interpreted as cosine distances."
        )
    if config["clustering"].get("splitting", False):
        raise ValueError("Splitting must remain disabled for this grid search.")
    if config["clustering"].get("max_clusters", 0) <= config["clustering"]["n_clusters"]:
        raise ValueError(
            "clustering.max_clusters must be greater than n_clusters when "
            "searching new-cluster thresholds."
        )

    chunks, questions, vectors, query_map = load_growth_data(config)
    experiment = config.get("experiment", {})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if args.output
        else config_path(config, experiment.get("output_root", "../experiments"))
        / f"{experiment.get('name', 'experiment')}_merge_grid_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    all_rows = []
    for new_threshold in args.new_cluster_thresholds:
        for merge_threshold in args.merge_thresholds:
            print(
                "Running new_cluster_threshold="
                f"{new_threshold:.4f}, merge_threshold={merge_threshold:.4f}",
                flush=True,
            )
            all_rows.extend(run_threshold(
                config, chunks, questions, vectors, query_map,
                new_threshold, merge_threshold,
            ))

    first_pair = (args.new_cluster_thresholds[0], args.merge_thresholds[0])
    reference = {
        row["stage"]: row
        for row in all_rows
        if (row["new_cluster_threshold"], row["merge_threshold"]) == first_pair
    }
    for row in all_rows:
        reference_row = reference[row["stage"]]
        row["delta_gold_hit_at_10_vs_first_threshold"] = (
            row["gold_hit_at_10"] - reference_row["gold_hit_at_10"]
        )
        row["delta_latency_ms_vs_first_threshold"] = (
            row["latency_ms"] - reference_row["latency_ms"]
        )
        row["delta_update_total_latency_ms_vs_first_threshold"] = (
            row["update_total_latency_ms"]
            - reference_row["update_total_latency_ms"]
        )

    best_params = select_best_params(all_rows, args.objective)
    save_rows(output / "grid_results.csv", all_rows)
    (output / "grid_results.json").write_text(
        json.dumps(all_rows, indent=2), encoding="utf-8"
    )
    (output / "best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    create_grid_plots(output / "plots", all_rows)
    print(f"\nResults saved to {output}")
    print(f"Plots saved to {output / 'plots'}")
    print(
        "\nBest parameters "
        f"(objective={best_params['objective']}, "
        f"stage={best_params['stage']}): "
        f"new_cluster_threshold={best_params['new_cluster_threshold']:.4f}, "
        f"merge_threshold={best_params['merge_threshold']:.4f}, "
        f"value={best_params['metrics'][args.objective]:.4f}"
    )
    print_table(all_rows)


if __name__ == "__main__":
    main()
