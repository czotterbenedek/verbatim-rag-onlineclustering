#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster_index import ClusteredRetrievalIndex
from src.config import config_path, load_config, write_config
from src.data_pipeline import read_jsonl
from src.evaluation import evaluate
from src.generation import OllamaGenerator
from scripts.run_evaluation import build_experiment_directory


def log(message):
    print(f"[full_chat] {message}", flush=True)


def _select_questions(questions, max_questions):
    if max_questions is None or max_questions >= len({row["question"] for row in questions}):
        return questions
    selected = set()
    for row in questions:
        if row["question"] not in selected:
            selected.add(row["question"])
            if len(selected) == max_questions:
                break
    return [row for row in questions if row["question"] in selected]


def _save_chat_run(directory, name, predictions, metrics):
    output = directory / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "chat_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (output / "chat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _generate_run(retrieval_predictions, chunks_by_id, generator, system_prompt, system_name):
    results = []
    total = len(retrieval_predictions)
    log(f"{system_name}: starting generation for {total} questions")
    for question_number, row in enumerate(retrieval_predictions, start=1):
        try:
            contexts = [chunks_by_id[chunk_id] for chunk_id in row["retrieved_chunks"]]
            answer, generation_latency_ms = generator.generate(
                row["question"], contexts, system_prompt
            )
        except Exception as error:
            log(
                f"{system_name}: failed at {question_number}/{total} "
                f"question={row['question']!r}: {error}"
            )
            raise
        result = dict(row)
        result.update({
            "answer": answer,
            "generation_latency_ms": generation_latency_ms,
            "full_pipeline_latency_ms": row["latency_ms"] + generation_latency_ms,
            "context_chunk_ids": row["retrieved_chunks"],
        })
        results.append(result)
        log(
            f"{system_name}: {question_number}/{total} | "
            f"retrieval={row['retrieval_latency_ms']:.1f} ms | "
            f"generation={generation_latency_ms:.1f} ms | "
            f"full_pipeline={result['full_pipeline_latency_ms']:.1f} ms"
        )
    log(f"{system_name}: completed generation for {total} questions")
    return results


def _chat_metrics(rows):
    if not rows:
        return {}
    latencies = np.asarray([row["full_pipeline_latency_ms"] for row in rows])
    generation = np.asarray([row["generation_latency_ms"] for row in rows])
    metrics = {
        "num_queries": len(rows),
        "mean_generation_latency_ms": float(generation.mean()),
        "mean_full_pipeline_latency_ms": float(latencies.mean()),
        "median_full_pipeline_latency_ms": float(np.median(latencies)),
        "p95_full_pipeline_latency_ms": float(np.percentile(latencies, 95)),
        "mean_retrieval_latency_ms": float(np.mean([row["retrieval_latency_ms"] for row in rows])),
        "mean_retrieval_fraction_of_pipeline": float(
            np.mean([row["retrieval_latency_ms"] / max(row["full_pipeline_latency_ms"], 1e-12)
                     for row in rows])
        ),
    }
    for field in ("retrieval_speedup_vs_baseline", "full_pipeline_speedup_vs_baseline"):
        values = [row[field] for row in rows if field in row]
        if values:
            metrics[f"mean_{field}"] = float(np.mean(values))
            metrics[f"median_{field}"] = float(np.median(values))
    return metrics


def _save_latency_plot(output_directory, chat_runs):
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    systems = list(chat_runs)
    labels = [system.replace("_", "\n").title() for system in systems]
    retrieval = np.asarray([
        np.mean([row["retrieval_latency_ms"] for row in chat_runs[system]])
        for system in systems
    ])
    generation = np.asarray([
        np.mean([row["generation_latency_ms"] for row in chat_runs[system]])
        for system in systems
    ])
    totals = np.asarray([
        np.mean([row["full_pipeline_latency_ms"] for row in chat_runs[system]])
        for system in systems
    ])
    positions = np.arange(len(systems))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions, retrieval, label="Retrieval", color="#3b82f6")
    axis.bar(positions, generation, bottom=retrieval, label="Generation", color="#f59e0b")
    for position, total in zip(positions, totals):
        axis.annotate(
            f"{total:.1f} ms",
            (position, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )
    axis.set(
        title="Average full-pipeline latency by system",
        xlabel="System",
        ylabel="Latency (ms)",
        xticks=positions,
        xticklabels=labels,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    path = output_directory / "full_pipeline_latency_by_system.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run the full Verbatim-RAG chat pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--name", default="full_chat")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    log(f"loading configuration from {args.config}")
    experiment_directory = build_experiment_directory(config, args.name)
    write_config(experiment_directory / "config.yaml", config)

    data, embedding = config["data"], config["embedding"]
    log("loading chunks, questions, and persisted embeddings")
    chunks = read_jsonl(config_path(config, data["chunks_path"]))
    questions = read_jsonl(config_path(config, data["evaluation_path"]))
    full_chat = config.get("full_chat", {})
    max_questions = args.max_questions if args.max_questions is not None else full_chat.get("max_questions", 100)
    questions = _select_questions(questions, max_questions)
    vectors = np.load(config_path(config, embedding["path"]))
    query_vectors = np.load(config_path(config, embedding["query_path"]))
    query_names = json.loads(config_path(config, embedding["query_path"]).with_suffix(".json").read_text())
    query_map = dict(zip(query_names, query_vectors))
    log(
        f"loaded {len(chunks):,} chunks, {len(questions):,} question rows, "
        f"{len(query_map):,} query embeddings; evaluating "
        f"{len({row['question'] for row in questions}):,} unique questions"
    )
    if vectors.shape[0] != len(chunks):
        raise ValueError("Embedding/chunk mismatch. Run scripts/create_embeddings.py again.")

    retrieval = config["retrieval"]
    clustering = config["clustering"]
    systems = config.get("experiment", {}).get("systems", [
        "baseline", "random", "offline_kmeans", "online_kmeans"
    ])
    log(f"retrieval systems: {', '.join(systems)}")
    retrieval_runs = {}
    if "baseline" in systems:
        log("baseline: starting retrieval evaluation")
        retrieval_runs["baseline"] = evaluate(
            questions, chunks, vectors, query_map, retrieval["top_k"], system_name="baseline"
        )
        log(f"baseline: retrieval evaluation complete ({len(retrieval_runs['baseline'])} questions)")

    for system_name, algorithm in (("offline_kmeans", "offline_kmeans"),
                                   ("online_kmeans", "online_kmeans")):
        if system_name not in systems:
            continue
        log(f"{system_name}: building retrieval index")
        index = ClusteredRetrievalIndex(
            vectors, clustering["n_clusters"], clustering["metric"],
            clustering["random_state"], clustering["batch_size"],
            clustering.get("algorithm", algorithm) if system_name == "online_kmeans" else algorithm,
            clustering.get("max_clusters"),
            clustering.get("new_cluster_threshold") if clustering.get("dynamic_creation", False) else None,
            clustering.get("merge_threshold") if clustering.get("merging", False) else None,
            clustering.get("split_conductance_threshold") if clustering.get("splitting", False) else None,
        )
        retrieval_runs[system_name] = evaluate(
            questions, chunks, vectors, query_map, retrieval["top_k"],
            index, clustering["top_clusters"], system_name=system_name,
        )
        log(f"{system_name}: retrieval evaluation complete ({len(retrieval_runs[system_name])} questions)")

    if "random" in systems:
        log("random: starting retrieval evaluation")
        random_config = config.get("random", {})
        fraction = random_config.get("candidate_fraction") or 0.1
        question_numbers = {q: n for n, q in enumerate(dict.fromkeys(row["question"] for row in questions))}

        def select_random(question, _query_embedding):
            count = max(1, min(len(chunks), int(len(chunks) * fraction)))
            rng = np.random.RandomState(random_config.get("random_state", 42) + question_numbers[question])
            return np.sort(rng.choice(len(chunks), count, replace=False)), []

        retrieval_runs["random"] = evaluate(
            questions, chunks, vectors, query_map, retrieval["top_k"],
            candidate_selector=select_random, system_name="random",
        )
        log(f"random: retrieval evaluation complete ({len(retrieval_runs['random'])} questions)")

    generator = OllamaGenerator(
        full_chat.get("model", "llama3.2:3b"),
        full_chat.get("endpoint", "http://localhost:11434/api/generate"),
        full_chat.get("timeout_seconds", 120),
    )
    log(f"generation model: {generator.model}; endpoint: {generator.endpoint}")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    system_prompt = full_chat.get(
        "system_prompt",
        "Answer the question using only the retrieved context. Be concise and say when the context is insufficient.",
    )
    chat_runs = {
        system_name: _generate_run(
            retrieval_predictions, chunks_by_id, generator, system_prompt, system_name
        )
        for system_name, retrieval_predictions in retrieval_runs.items()
    }
    baseline = chat_runs.get("baseline")
    if baseline is not None:
        baseline_by_question = {row["question"]: row for row in baseline}
        for system_name, chat_predictions in chat_runs.items():
            if system_name == "baseline":
                continue
            for row in chat_predictions:
                baseline_row = baseline_by_question.get(row["question"])
                if baseline_row is not None:
                    row["retrieval_speedup_vs_baseline"] = (
                        baseline_row["retrieval_latency_ms"]
                        / max(row["retrieval_latency_ms"], 1e-12)
                    )
                    row["full_pipeline_speedup_vs_baseline"] = (
                        baseline_row["full_pipeline_latency_ms"]
                        / max(row["full_pipeline_latency_ms"], 1e-12)
                    )
    for system_name, chat_predictions in chat_runs.items():
        _save_chat_run(experiment_directory, system_name, chat_predictions, _chat_metrics(chat_predictions))
    plot_path = _save_latency_plot(experiment_directory / "plots", chat_runs)
    log(f"latency plot saved to {plot_path}")

    metadata = {
        "experiment_mode": "full_chat",
        "num_questions": len({row["question"] for row in questions}),
        "model": generator.model,
        "systems": list(retrieval_runs),
    }
    (experiment_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Full-chat experiment saved to {experiment_directory}")


if __name__ == "__main__":
    main()
