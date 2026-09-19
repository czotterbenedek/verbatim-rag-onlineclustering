#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster_index import ClusteredRetrievalIndex
from src.config import config_path, load_config, write_config
from src.data_pipeline import read_jsonl
from src.evaluation import evaluate, summarize
from src.plots import create_plots


def save_run(directory, name, predictions):
    output = directory / name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    metrics = summarize(predictions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"predictions": predictions, "metrics": metrics}


def build_experiment_directory(config, requested_name):
    experiment = config.get("experiment", {})
    name = requested_name or experiment.get("name", "experiment")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = config_path(config, experiment.get("output_root", "../experiments"))
    directory = root / f"{name}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def main():
    parser = argparse.ArgumentParser(description="Run configured Verbatim-RAG experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", help="Optional experiment name; timestamp is always appended.")
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_directory = build_experiment_directory(config, args.name)
    write_config(experiment_directory / "config.yaml", config)

    data, embedding = config["data"], config["embedding"]
    chunks = read_jsonl(config_path(config, data["chunks_path"]))
    questions = read_jsonl(config_path(config, data["evaluation_path"]))
    vectors = np.load(config_path(config, embedding["path"]))
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding/chunk mismatch: {vectors.shape[0]} embeddings for {len(chunks)} chunks. "
            "Run scripts/create_embeddings.py again after preparing data."
        )
    query_vectors = np.load(config_path(config, embedding["query_path"]))
    query_names = json.loads(config_path(config, embedding["query_path"]).with_suffix(".json").read_text(encoding="utf-8"))
    query_map = dict(zip(query_names, query_vectors))
    retrieval = config["retrieval"]
    clustering = config["clustering"]
    systems = config.get("experiment", {}).get("systems", ["baseline", "online_kmeans"])
    runs = {}

    if "baseline" in systems:
        predictions = evaluate(questions, chunks, vectors, query_map, retrieval["top_k"], system_name="baseline")
        runs["baseline"] = save_run(experiment_directory, "baseline", predictions)

    indexes = {}
    for system_name, algorithm in (("offline_kmeans", "offline_kmeans"), ("online_kmeans", "online_kmeans")):
        if system_name in systems:
            indexes[system_name] = ClusteredRetrievalIndex(
                vectors, clustering["n_clusters"], clustering["metric"],
                clustering["random_state"], clustering["batch_size"],
                clustering.get("algorithm", algorithm) if system_name == "online_kmeans" else algorithm,
                clustering.get("max_clusters"),
                clustering.get("new_cluster_threshold") if clustering.get("dynamic_creation", False) else None,
                clustering.get("merge_threshold") if clustering.get("merging", False) else None,
                clustering.get("split_conductance_threshold") if clustering.get("splitting", False) else None,
            )
            predictions = evaluate(
                questions, chunks, vectors, query_map, retrieval["top_k"],
                indexes[system_name], clustering["top_clusters"], system_name=system_name,
            )
            runs[system_name] = save_run(experiment_directory, system_name, predictions)

    if "random" in systems:
        random_config = config.get("random", {})
        target_fraction = random_config.get("candidate_fraction")
        reference = runs.get("online_kmeans", {}).get("predictions", [])
        target_counts = {row["question"]: row["num_candidates"] for row in reference}
        if target_fraction is not None:
            target_counts = {row["question"]: max(1, int(len(chunks) * target_fraction)) for row in questions}
        question_numbers = {question: number for number, question in enumerate(dict.fromkeys(row["question"] for row in questions))}
        seed = random_config.get("random_state", clustering["random_state"])

        def select_random(question, _query_embedding):
            count = min(len(chunks), target_counts.get(question, max(1, len(chunks) // 10)))
            rng = np.random.RandomState(seed + question_numbers[question])
            return np.sort(rng.choice(len(chunks), count, replace=False)), []

        predictions = evaluate(
            questions, chunks, vectors, query_map, retrieval["top_k"],
            candidate_selector=select_random, system_name="random",
        )
        runs["random"] = save_run(experiment_directory, "random", predictions)

    for system_name, run in runs.items():
        if system_name == "baseline":
            continue
        baseline_by_question = {row["question"]: set(row["retrieved_chunks"])
                                for row in runs.get("baseline", {}).get("predictions", [])}
        baseline_latency_by_question = {
            row["question"]: row["latency_ms"]
            for row in runs.get("baseline", {}).get("predictions", [])
        }
        baseline_retrieval_latency_by_question = {
            row["question"]: row["retrieval_latency_ms"]
            for row in runs.get("baseline", {}).get("predictions", [])
        }
        for row in run["predictions"]:
            baseline_ids = baseline_by_question.get(row["question"], set())
            row["baseline_overlap_at_k"] = len(baseline_ids.intersection(row["retrieved_chunks"])) / max(1, len(baseline_ids))
            row["baseline_latency_ms"] = baseline_latency_by_question.get(row["question"])
            row["baseline_retrieval_latency_ms"] = baseline_retrieval_latency_by_question.get(row["question"])
            if row["baseline_latency_ms"] is not None:
                row["total_speedup_vs_baseline"] = row["baseline_latency_ms"] / max(row["latency_ms"], 1e-12)
                row["total_latency_saved_ms"] = row["baseline_latency_ms"] - row["latency_ms"]
            if row["baseline_retrieval_latency_ms"] is not None:
                row["retrieval_speedup_vs_baseline"] = row["baseline_retrieval_latency_ms"] / max(row["retrieval_latency_ms"], 1e-12)
                row["retrieval_latency_saved_ms"] = row["baseline_retrieval_latency_ms"] - row["retrieval_latency_ms"]
        run["metrics"] = summarize(run["predictions"])
        run["metrics"]["num_clusters"] = (clustering["n_clusters"]
                            if system_name in ("offline_kmeans", "online_kmeans") else 0)
        run["metrics"]["top_clusters"] = (clustering["top_clusters"]
                            if system_name in ("offline_kmeans", "online_kmeans") else 0)
        run_directory = experiment_directory / system_name
        (run_directory / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in run["predictions"]), encoding="utf-8"
        )
        (run_directory / "metrics.json").write_text(json.dumps(run["metrics"], indent=2), encoding="utf-8")

    create_plots(experiment_directory / "plots", runs)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "systems": list(runs),
        "num_chunks": len(chunks),
        "num_documents": len({chunk["paper_id"] for chunk in chunks}),
        "num_queries": len(dict.fromkeys(row["question"] for row in questions)),
        "embedding_dimension": int(vectors.shape[1]),
        "evaluation_split": data["evaluation_split"],
        "gold_paper_coverage": len({chunk["paper_id"] for chunk in chunks}.intersection(
            {str(row["gold_paper"]) for row in questions if row.get("gold_paper")}
        )),
    }
    (experiment_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Experiment saved to {experiment_directory}")


if __name__ == "__main__":
    main()
