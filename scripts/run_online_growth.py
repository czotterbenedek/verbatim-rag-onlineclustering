#!/usr/bin/env python
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster_index import ClusteredRetrievalIndex
from src.config import config_path, load_config, write_config
from src.data_pipeline import read_jsonl
from src.evaluation import evaluate, summarize
from src.plots import create_growth_plots


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(message):
    print(f"[online-growth] {message}", flush=True)


def build_directory(config, requested_name):
    experiment = config.get("experiment", {})
    name = requested_name or f"{experiment.get('name', 'experiment')}_online_growth"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = config_path(config, experiment.get("output_root", "../experiments")) / f"{name}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def stage_endpoints(total_chunks, fractions):
    endpoints = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"Growth stages must be in (0, 1], got {fraction}.")
        endpoint = max(1, min(total_chunks, int(np.ceil(total_chunks * fraction))))
        if not endpoints or endpoint > endpoints[-1]:
            endpoints.append(endpoint)
    if not endpoints or endpoints[-1] != total_chunks:
        endpoints.append(total_chunks)
    return endpoints


def add_baseline_comparisons(predictions, baseline_predictions):
    baseline_by_question = {row["question"]: row for row in baseline_predictions}
    for row in predictions:
        baseline = baseline_by_question.get(row["question"])
        if baseline is None:
            continue
        baseline_ids = set(baseline["retrieved_chunks"])
        row["baseline_overlap_at_k"] = len(baseline_ids.intersection(row["retrieved_chunks"])) / max(1, len(baseline_ids))
        row["baseline_latency_ms"] = baseline["latency_ms"]
        row["baseline_retrieval_latency_ms"] = baseline["retrieval_latency_ms"]
        row["retrieval_speedup_vs_baseline"] = baseline["retrieval_latency_ms"] / max(row["retrieval_latency_ms"], 1e-12)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run a growing-corpus Online K-Means experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", help="Optional experiment name; timestamp is always appended.")
    args = parser.parse_args()
    config = load_config(args.config)
    log(f"Loading configuration from {args.config}")
    data, embedding = config["data"], config["embedding"]
    growth = config.get("online_growth", {})
    chunks = read_jsonl(config_path(config, data["chunks_path"]))
    questions = read_jsonl(config_path(config, data["evaluation_path"]))
    vectors = np.load(config_path(config, embedding["path"]))
    log(f"Loaded {len(chunks):,} chunks, {len(questions):,} evaluation rows, and embeddings {vectors.shape}")
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding/chunk mismatch: {vectors.shape[0]} embeddings for {len(chunks)} chunks. "
            "Run scripts/create_embeddings.py again after preparing data."
        )
    query_path = config_path(config, embedding["query_path"])
    query_vectors = np.load(query_path)
    query_names = json.loads(query_path.with_suffix(".json").read_text(encoding="utf-8"))
    query_map = dict(zip(query_names, query_vectors))
    log(f"Loaded {len(query_map):,} persisted query embeddings")
    if growth.get("order", "fixed") == "shuffled":
        rng = np.random.RandomState(growth.get("stream_seed", 42))
        order = rng.permutation(len(chunks))
    elif growth.get("order", "fixed") == "fixed":
        order = np.arange(len(chunks))
    else:
        raise ValueError("online_growth.order must be 'fixed' or 'shuffled'.")

    directory = build_directory(config, args.name)
    write_config(directory / "config.yaml", config)
    ordered_chunks = [chunks[int(index)] for index in order]
    ordered_vectors = vectors[order]
    clustering = config["clustering"]
    systems = growth.get("systems", ["baseline", "offline_kmeans", "online_kmeans"])
    if "online_kmeans" not in systems:
        raise ValueError("online_growth.systems must include online_kmeans.")
    endpoints = stage_endpoints(len(ordered_chunks), growth.get("stages", [1.0]))
    log(f"Stream order: {growth.get('order', 'fixed')}; checkpoints: {endpoints}")
    log(f"Systems: {', '.join(systems)}; output: {directory}")
    if endpoints[0] < clustering["n_clusters"]:
        raise ValueError("The first growth stage must contain at least n_clusters embeddings.")

    index = None
    updates = []
    stage_records = []
    previous_end = 0
    for stage_number, stage_end in enumerate(endpoints):
        log(f"Stage {stage_number + 1}/{len(endpoints)}: growing from {previous_end:,} to {stage_end:,} chunks")
        stage_chunks = ordered_chunks[:stage_end]
        stage_vectors = ordered_vectors[:stage_end]
        new_vectors = ordered_vectors[previous_end:stage_end]
        visible_papers = {str(chunk["paper_id"]) for chunk in stage_chunks}
        stage_questions = [
            row for row in questions
            if str(row.get("gold_paper")) in visible_papers
        ]
        log(f"Stage {stage_number + 1}: {len(new_vectors):,} new chunks, {len(stage_questions):,} available questions")
        update_started = time.perf_counter()
        if index is None:
            log("Building initial Online K-Means index")
            index = ClusteredRetrievalIndex(
                new_vectors, clustering["n_clusters"], clustering["metric"],
                clustering["random_state"], growth.get("update_batch_size", clustering["batch_size"]),
                "online_kmeans", clustering.get("max_clusters"),
                clustering.get("new_cluster_threshold") if clustering.get("dynamic_creation", False) else None,
                clustering.get("merge_threshold") if clustering.get("merging", False) else None,
                clustering.get("split_conductance_threshold") if clustering.get("splitting", False) else None,
            )
            initial_latency_ms = (time.perf_counter() - update_started) * 1000
            update = {
                "update_latency_ms": initial_latency_ms,
                "assignment_latency_ms": 0.0,
                "total_latency_ms": initial_latency_ms,
                "throughput_chunks_per_second": stage_end / max(initial_latency_ms / 1000, 1e-12),
                "centroid_drift_mean": 0.0,
                "num_clusters": int(len(index.kmeans.centroids)),
                "total_seen": int(index.kmeans.total_seen),
            }
        else:
            log("Updating Online K-Means index")
            update = index.append_embeddings(new_vectors, growth.get("update_batch_size", clustering["batch_size"]))
        log(f"Online update complete in {update['total_latency_ms']:.1f} ms ({update['num_clusters']} clusters)")

        predictions_by_system = {}
        if "baseline" in systems:
            log("Evaluating baseline retrieval")
            predictions_by_system["baseline"] = evaluate(
                stage_questions, stage_chunks, stage_vectors, query_map,
                config["retrieval"]["top_k"], system_name="baseline"
            )

        offline_fit_latency_ms = None
        if "offline_kmeans" in systems:
            log("Fitting offline K-Means from scratch")
            offline_started = time.perf_counter()
            offline_index = ClusteredRetrievalIndex(
                stage_vectors, clustering["n_clusters"], clustering["metric"],
                clustering["random_state"], clustering["batch_size"], "offline_kmeans",
            )
            offline_fit_latency_ms = (time.perf_counter() - offline_started) * 1000
            log(f"Offline K-Means fit complete in {offline_fit_latency_ms:.1f} ms")
            log("Evaluating offline K-Means retrieval")
            predictions_by_system["offline_kmeans"] = evaluate(
                stage_questions, stage_chunks, stage_vectors, query_map,
                config["retrieval"]["top_k"], offline_index,
                clustering["top_clusters"], system_name="offline_kmeans",
            )

        log("Evaluating online K-Means retrieval")
        predictions_by_system["online_kmeans"] = evaluate(
            stage_questions, stage_chunks, stage_vectors, query_map,
            config["retrieval"]["top_k"],
            index=index, top_clusters=clustering["top_clusters"], system_name="online_kmeans",
        )
        baseline_predictions = predictions_by_system.get("baseline", [])
        for system_name, predictions in predictions_by_system.items():
            if system_name != "baseline" and baseline_predictions:
                predictions_by_system[system_name] = add_baseline_comparisons(
                    predictions, baseline_predictions
                )
        stage_directory = directory / "stages" / f"stage_{stage_number:03d}"
        stage_metrics = {}
        for system_name, predictions in predictions_by_system.items():
            save_jsonl(stage_directory / system_name / "predictions.jsonl", predictions)
            metrics = summarize(predictions)
            metrics["num_clusters"] = (
                0 if system_name == "baseline" else clustering["n_clusters"]
            )
            stage_metrics[system_name] = metrics
            (stage_directory / system_name / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            log(f"{system_name}: Hit@10={metrics.get('gold_hit_at_10', 0.0):.1%}, "
                f"candidate fraction={metrics.get('candidate_fraction', 0.0):.1%}")
        update_record = {
            "stage": stage_number,
            "num_questions": len(stage_questions),
            "num_new_chunks": stage_end - previous_end,
            "num_chunks_before": previous_end,
            "num_chunks_after": stage_end,
            "offline_fit_latency_ms": offline_fit_latency_ms,
            **update,
        }
        updates.append(update_record)
        stage_records.append({
            "stage": stage_number,
            "num_chunks": stage_end,
            "update": update_record,
            "metrics": stage_metrics,
        })
        previous_end = stage_end
        log(f"Stage {stage_number + 1} outputs saved to {stage_directory}")

    log("Writing update log and longitudinal plots")
    save_jsonl(directory / "updates.jsonl", updates)
    create_growth_plots(directory / "plots", stage_records)
    (directory / "metadata.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": "online_growth",
        "num_chunks": len(chunks),
        "num_stages": len(endpoints),
        "stream_order": growth.get("order", "fixed"),
        "stream_seed": growth.get("stream_seed", 42),
        "stages": stage_records,
    }, indent=2), encoding="utf-8")
    log(f"Experiment saved to {directory}")


if __name__ == "__main__":
    main()