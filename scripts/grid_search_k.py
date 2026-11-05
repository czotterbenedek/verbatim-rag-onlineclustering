#!/usr/bin/env python
"""Sweep the K-Means cluster count and create an elbow plot."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster_index import ClusteredRetrievalIndex
from src.config import config_path, load_config


def parse_k_values(value):
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            k = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid cluster count {item!r}; expected a positive integer."
            ) from exc
        if k <= 0:
            raise argparse.ArgumentTypeError("Cluster counts must be positive.")
        values.append(k)
    if not values:
        raise argparse.ArgumentTypeError("At least one cluster count is required.")
    return sorted(set(values))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clustering_objective(embeddings, centroids, labels, metric):
    points = np.asarray(embeddings, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)
    assigned = centers[np.asarray(labels, dtype=int)]

    if metric == "cosine":
        point_norms = np.linalg.norm(points, axis=1, keepdims=True)
        center_norms = np.linalg.norm(assigned, axis=1, keepdims=True)
        normalized_points = points / np.maximum(point_norms, 1e-12)
        normalized_centers = assigned / np.maximum(center_norms, 1e-12)
        distances = 1.0 - np.sum(normalized_points * normalized_centers, axis=1)
    elif metric == "euclidean":
        distances = np.linalg.norm(points - assigned, axis=1)
    else:
        raise ValueError(f"Unsupported clustering metric: {metric}")

    return float(np.sum(distances ** 2))


def create_elbow_plot(output_directory, rows, metric):
    import matplotlib.pyplot as plt

    plot_directory = output_directory / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    positions = [row["k"] for row in rows]
    objective = [row["objective"] for row in rows]
    normalized = [row["objective_per_embedding"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(positions, objective, marker="o", linewidth=2)
    axes[0].set(title=f"K-Means elbow ({metric} objective)", xlabel="Number of clusters (k)",
                ylabel="Sum of squared distances")
    axes[1].plot(positions, normalized, marker="o", linewidth=2)
    axes[1].set(title="Normalized elbow", xlabel="Number of clusters (k)",
                ylabel="Objective per embedding")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(plot_directory / "elbow_plot.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep K-Means cluster counts and create an elbow diagram."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=parse_k_values("10,25,50,75,100,150,200,300,400,500"),
        help="Comma-separated cluster counts to evaluate.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("offline_kmeans", "online_kmeans"),
        default="offline_kmeans",
        help="K-Means implementation to fit for the sweep.",
    )
    parser.add_argument("--output", help="Output directory for sweep results.")
    args = parser.parse_args()

    config = load_config(args.config)
    embedding = config["embedding"]
    clustering = config["clustering"]
    embeddings = np.load(config_path(config, embedding["path"]))
    objective_metric = "euclidean" if args.algorithm == "offline_kmeans" else clustering["metric"]

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")
    if max(args.k_values) > len(embeddings):
        raise ValueError(
            f"Largest k ({max(args.k_values)}) cannot exceed the number of "
            f"embeddings ({len(embeddings)})."
        )

    if args.output:
        output_directory = Path(args.output)
        if not output_directory.is_absolute():
            output_directory = config_path(config, args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_directory = config_path(config, config["experiment"]["output_root"]) / (
            f"k_search_{args.algorithm}_{timestamp}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in tqdm(args.k_values, desc=f"{args.algorithm} k sweep", unit="k"):
        tqdm.write(f"[k-search] fitting {args.algorithm} with k={k}")
        started = time.perf_counter()
        index = ClusteredRetrievalIndex(
            embeddings,
            k,
            clustering["metric"],
            clustering["random_state"],
            clustering["batch_size"],
            args.algorithm,
            k if args.algorithm == "online_kmeans" else None,
        )
        objective = clustering_objective(
            embeddings,
            index.kmeans.centroids
            if args.algorithm == "online_kmeans"
            else index.kmeans.cluster_centers_,
            index.chunk_to_cluster,
            objective_metric,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows.append({
            "k": k,
            "objective": objective,
            "objective_per_embedding": objective / len(embeddings),
            "num_clusters": len(index.cluster_to_chunks),
            "fit_time_ms": elapsed_ms,
        })

    save_rows(output_directory / "elbow_results.csv", rows)
    (output_directory / "elbow_results.json").write_text(
        json.dumps({
            "algorithm": args.algorithm,
            "metric": objective_metric,
            "embedding_count": len(embeddings),
            "k_values": args.k_values,
            "results": rows,
        }, indent=2),
        encoding="utf-8",
    )
    create_elbow_plot(output_directory, rows, objective_metric)
    print(f"[k-search] results written to {output_directory}", flush=True)


if __name__ == "__main__":
    main()
