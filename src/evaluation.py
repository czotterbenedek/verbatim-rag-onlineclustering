from collections import defaultdict
import time

import numpy as np

from .retrieval import VerbatimRAGRetriever


def _gold_ids(rows, chunks):
    chunks_by_paper = defaultdict(list)
    for chunk in chunks:
        chunks_by_paper[chunk["paper_id"]].append(chunk)
    result = set()
    for row in rows:
        paper = row.get("gold_paper") or row.get("paper_id")
        chunk = row.get("gold_chunk") if row.get("gold_chunk") is not None else row.get("chunk_index")
        if paper is None:
            continue
        span_texts = (row.get("spans") or {}).get("text", [])
        if isinstance(span_texts, str):
            span_texts = [span_texts]
        matched = False
        for candidate in chunks_by_paper.get(str(paper), []):
            if any(span and span in candidate["text"] for span in span_texts):
                result.add(candidate["chunk_id"])
                matched = True
        if not matched and chunk is not None:
            fallback_id = f"{paper}_{chunk}"
            if any(item["chunk_id"] == fallback_id for item in chunks_by_paper.get(str(paper), [])):
                result.add(fallback_id)
    return result


def evaluate(questions, chunks, embeddings, query_embeddings, top_k, index=None,
             top_clusters=10, candidate_selector=None, system_name="baseline"):
    grouped = defaultdict(list)
    for row in questions:
        grouped[row["question"]].append(row)
    positions = {chunk["chunk_id"]: position for position, chunk in enumerate(chunks)}
    retriever = VerbatimRAGRetriever(chunks, embeddings, query_embeddings)
    results = []
    for query_number, (question, rows) in enumerate(grouped.items()):
        embedding_started = time.perf_counter()
        query_embedding = query_embeddings[question]
        embedding_latency_ms = (time.perf_counter() - embedding_started) * 1000
        gold = _gold_ids(rows, chunks)
        started = time.perf_counter()
        if index is None:
            if candidate_selector is None:
                candidate_ids, selected_clusters = None, []
            else:
                candidate_ids, selected_clusters = candidate_selector(question, query_embedding)
            routing_latency_ms = 0.0
            candidate_construction_latency_ms = 0.0
        else:
            routing_started = time.perf_counter()
            selected_clusters, _ = index.predict_top_clusters(query_embedding, top_clusters)
            routing_latency_ms = (time.perf_counter() - routing_started) * 1000
            construction_started = time.perf_counter()
            candidate_ids = index.candidate_ids_from_clusters(selected_clusters)
            candidate_construction_latency_ms = (time.perf_counter() - construction_started) * 1000
        retrieval_started = time.perf_counter()
        retrieved_ids, scores = retriever.search(question, top_k, candidate_ids)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        routing_hit = (None if index is None else any(
            index.chunk_to_cluster[positions[gold_id]] in selected_clusters
            for gold_id in gold if gold_id in positions
        ))
        results.append({
            "query_id": str(query_number),
            "system": system_name,
            "question": question,
            "gold_chunk_ids": sorted(gold),
            "selected_clusters": selected_clusters.tolist() if hasattr(selected_clusters, "tolist") else selected_clusters,
            "num_candidates": int(len(candidate_ids) if candidate_ids is not None else len(chunks)),
            "candidate_fraction": float((len(candidate_ids) if candidate_ids is not None else len(chunks)) / max(1, len(chunks))),
            "retrieved_chunks": retrieved_ids,
            "retrieval_scores": scores,
            "routing_hit": None if routing_hit is None else bool(routing_hit),
            "embedding_latency_ms": embedding_latency_ms,
            "routing_latency_ms": routing_latency_ms,
            "candidate_construction_latency_ms": candidate_construction_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "latency_ms": (time.perf_counter() - started) * 1000,
            **{f"gold_hit_at_{k}": bool(gold.intersection(retrieved_ids[:k])) for k in (1, 3, 5, 10)},
        })
    return results


def summarize(results):
    if not results:
        return {}
    fields = ("candidate_fraction", "latency_ms", "embedding_latency_ms", "routing_latency_ms",
              "candidate_construction_latency_ms", "retrieval_latency_ms", "routing_hit",
              "gold_hit_at_1", "gold_hit_at_3", "gold_hit_at_5", "gold_hit_at_10")
    metrics = {field: float(np.mean([row[field] for row in results]))
               for field in fields if field != "routing_hit"}
    routing_values = [row["routing_hit"] for row in results if row["routing_hit"] is not None]
    metrics["routing_hit"] = (float(np.mean(routing_values)) if routing_values else None)
    for field in ("latency_ms", "retrieval_latency_ms"):
        values = [row[field] for row in results]
        metrics[f"{field}_median"] = float(np.median(values))
        metrics[f"{field}_p95"] = float(np.percentile(values, 95))
    baseline_latencies = [row.get("baseline_latency_ms") for row in results]
    baseline_retrieval_latencies = [row.get("baseline_retrieval_latency_ms") for row in results]
    if all(value is not None for value in baseline_latencies):
        speedups = [baseline / max(row["latency_ms"], 1e-12)
                    for baseline, row in zip(baseline_latencies, results)]
        savings = [baseline - row["latency_ms"]
                   for baseline, row in zip(baseline_latencies, results)]
        metrics["total_speedup_vs_baseline"] = float(np.mean(speedups))
        metrics["total_speedup_median_vs_baseline"] = float(np.median(speedups))
        metrics["total_speedup_p95_vs_baseline"] = float(np.percentile(speedups, 95))
        metrics["mean_total_latency_saved_ms"] = float(np.mean(savings))
    if all(value is not None for value in baseline_retrieval_latencies):
        speedups = [baseline / max(row["retrieval_latency_ms"], 1e-12)
                    for baseline, row in zip(baseline_retrieval_latencies, results)]
        savings = [baseline - row["retrieval_latency_ms"]
                   for baseline, row in zip(baseline_retrieval_latencies, results)]
        metrics["retrieval_speedup_vs_baseline"] = float(np.mean(speedups))
        metrics["retrieval_speedup_median_vs_baseline"] = float(np.median(speedups))
        metrics["retrieval_speedup_p95_vs_baseline"] = float(np.percentile(speedups, 95))
        metrics["mean_retrieval_latency_saved_ms"] = float(np.mean(savings))
    if "baseline_overlap_at_k" in results[0]:
        metrics["baseline_overlap_at_k"] = float(np.mean([row["baseline_overlap_at_k"] for row in results]))
    metrics["num_queries"] = len(results)
    metrics["candidate_reduction"] = 1.0 - metrics["candidate_fraction"]
    return metrics