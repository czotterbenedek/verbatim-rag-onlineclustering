# Implementation Plan — Online K-Means + Verbatim-RAG

## 1. Research objective

Extend the existing Verbatim-RAG retrieval pipeline with an **online clustering-based candidate-selection layer**.

The research question is:

> **Can dynamically maintained Online K-Means clusters reduce the amount of the document collection that Verbatim-RAG has to search while preserving retrieval quality and evidence recall?**

The clustering layer should act as a **coarse semantic routing/indexing mechanism**.

The original Verbatim-RAG retrieval mechanism remains responsible for final retrieval.

### Core design principle

```text
Online K-Means = candidate generation / routing
Verbatim-RAG   = final retrieval
```

Do **not** replace the original Verbatim-RAG retrieval with centroid retrieval.

---

## 2. High-level architecture

```text
                         ACL Anthology
                              │
                              ▼
                        Document chunks
                              │
                              ▼
                         Embeddings
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Original embedding        Online K-Means
              index                     index
                                           │
                                      centroids
                                           │
                                           │
Query ──► Query embedding ────────────────┤
                                           ▼
                                  Top-N clusters
                                           │
                                           ▼
                                  Candidate chunks
                                           │
                                           ▼
                              Verbatim-RAG retrieval
                                           │
                                           ▼
                                   Top-K chunks
                                           │
                                           ▼
                               Verbatim span extraction
                                           │
                                           ▼
                                      Evidence
```

---

## 3. Dataset

Use the official ACL-Verbatim resources.

### 3.1 Evaluation dataset

Use:

`KRLabsOrg/acl-verbatim-spans`

specifically the `canonical` configuration.

The dataset currently contains:

| Split | Rows | Role |
|---|---:|---|
| train | 20,916 | Silver development/training |
| validation | 2,319 | Silver validation |
| test | 100 | Gold evaluation |

The canonical fields include:

```text
question
paper_id
chunk_index
chunk
label
answerable
spans
source
retrieval_rank
gold_paper
gold_chunk
```

The test set contains:

```text
20 unique questions
5 candidate chunks/question
100 total rows
47 relevant chunks
78 gold evidence spans
53 irrelevant chunks
```

The gold test set must not be used for parameter tuning.

Example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "KRLabsOrg/acl-verbatim-spans",
    "canonical"
)

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]
```

---

## 4. Underlying document corpus

Use the ACL Anthology Markdown corpus:

`KRLabsOrg/acl-anthology-md`

The full corpus is large, so use a staged approach.

### Phase A — development

Start with approximately:

```text
1,000–5,000 papers
```

Purpose:

- implement the pipeline
- debug indexing
- debug cluster assignment
- debug query routing
- test Verbatim-RAG integration
- measure memory/runtime
- validate evaluation

### Phase B — experiment

Move to approximately:

```text
10,000–20,000 papers
```

Purpose:

- parameter experiments
- cluster-count experiments
- routing experiments
- latency measurements
- ablation experiments

### Phase C — final experiment

Only after the pipeline is stable:

```text
full ACL corpus
```

The final experiment should demonstrate scalability.

---

## 5. Dataset split policy

The **gold test set must never be used to tune clustering parameters**.

Use:

```text
train
  ↓
development / parameter exploration

validation
  ↓
parameter selection

test
  ↓
final result only
```

The test set is small, so overfitting to it would be particularly easy.

---

## 6. Chunking

Use the chunking strategy already used by the existing Verbatim-RAG pipeline where possible.

Do not introduce a completely different chunk representation in the first experiment.

Every chunk should have a stable identifier.

Recommended metadata:

```text
chunk_id
paper_id
chunk_index
text
embedding
cluster_id
```

Recommended representation:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str
    embedding: np.ndarray
    cluster_id: int | None
```

The `chunk_id` should be deterministic, for example:

```text
{paper_id}_{chunk_index}
```

---

## 7. Embeddings

Use the same embedding model as the existing Verbatim-RAG setup where practical.

For the first implementation, do not introduce an embedding-model comparison.

The experiment should isolate:

```text
Original retrieval
        vs
Original retrieval + Online K-Means routing
```

Store embeddings so they do not have to be recomputed for every experiment.

Example:

```text
artifacts/
    embeddings/
        acl_embeddings.npy
    metadata/
        chunks.parquet
```

---

## 8. Online K-Means index

Reuse the existing `OnlineKMeans` implementation.

The implementation already supports:

- Euclidean distance
- cosine distance
- incremental updates
- dynamic cluster creation
- cluster merging
- cluster splitting
- centroid counts/statistics

For the **first experiment**:

```text
dynamic creation = OFF
merging          = OFF
splitting        = OFF
```

Start with a simple fixed clustering configuration.

---

## 9. Retrieval index abstraction

Do not put all clustering logic directly inside Verbatim-RAG.

Create an abstraction conceptually like:

```python
class ClusteredRetrievalIndex:
    """Use OnlineKMeans to select candidate clusters and
    delegate final retrieval to the existing Verbatim-RAG retriever.
    """
```

It should contain:

```text
embeddings
chunk metadata
cluster assignments
cluster → chunk mapping
OnlineKMeans instance
```

For example:

```python
cluster_to_chunks: dict[int, np.ndarray]
chunk_to_cluster: np.ndarray
```

---

## 10. Building the cluster index

Fit Online K-Means incrementally:

```python
kmeans = OnlineKMeans(
    n_clusters=K,
    metric="cosine",
    random_state=42,
)

for batch in embedding_batches:
    kmeans.partial_fit(batch)
```

After clustering, assign every chunk to its nearest centroid:

```python
cluster_ids = kmeans.predict(embeddings)
```

Build:

```python
cluster_to_chunks
```

Example:

```text
cluster 0 → [12, 19, 35, 91, ...]
cluster 1 → [2, 8, 14, ...]
...
```

---

## 11. Query-time routing

For every query:

```text
query
 ↓
query embedding
 ↓
distance to all centroids
 ↓
sort centroids
 ↓
select top-N clusters
```

Add a method such as:

```python
def predict_top_clusters(
    self,
    X,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Return:

```text
cluster IDs
cluster distances
```

Example:

```python
cluster_ids, distances = index.predict_top_clusters(
    query_embedding,
    n_clusters=10,
)
```

---

## 12. Candidate generation

After selecting clusters:

```python
candidate_chunk_ids = union(
    cluster_to_chunks[c]
    for c in selected_clusters
)
```

Pass the final candidate set to the existing Verbatim-RAG retrieval system.

```text
query_embedding
        ↓
top 10 clusters
        ↓
candidate_chunk_ids
        ↓
Verbatim-RAG
        ↓
top K retrieved chunks
```

---

## 13. Baseline system

Implement and freeze:

```text
Baseline:
Query
 ↓
original Verbatim-RAG retrieval
 ↓
top-K results
```

Record for every query:

```python
{
    "query_id": ...,
    "query": ...,
    "retrieved_chunk_ids": ...,
    "retrieval_scores": ...,
    "latency": ...
}
```

Do not change the baseline while experimenting with clustering.

---

## 14. Clustered system

Implement:

```text
Clustered:
Query
 ↓
query embedding
 ↓
Online K-Means
 ↓
top-N clusters
 ↓
candidate chunks
 ↓
original Verbatim-RAG retrieval
 ↓
top-K results
```

Record:

```python
{
    "selected_clusters": ...,
    "num_candidates": ...,
    "candidate_fraction": ...,
    "routing_latency": ...,
    "retrieval_latency": ...
}
```

---

## 15. Primary research variables

### Number of clusters searched

Test configurations such as:

```text
top-1
top-2
top-5
top-10
top-20
top-50
```

For each configuration calculate:

```text
candidate chunks
candidate fraction
retrieval quality
latency
```

### Total number of clusters

Starting values:

```text
K = 50
K = 100
K = 250
K = 500
K = 1000
K = 2000
```

These are starting points, not fixed requirements.

Report:

```text
total clusters
        ×
clusters searched
        ×
candidate corpus fraction
        ×
retrieval quality
```

---

## 16. Primary evaluation: gold evidence retrieval

The main question is:

> Does the clustered system still retrieve the chunk containing the gold evidence?

Use:

```text
paper_id
chunk_index
label
spans
gold_paper
gold_chunk
```

Calculate:

```text
Hit@1
Hit@3
Hit@5
Hit@10
```

A hit occurs when an answer-bearing/gold chunk appears in the final retrieved results.

---

## 17. Cluster-routing recall

Before Verbatim-RAG performs final retrieval:

> Was the gold chunk's cluster selected?

For each query:

```text
gold chunk
 ↓
its cluster
 ↓
was that cluster among top-N selected clusters?
```

Calculate:

\[
RoutingRecall@N =
rac{\#	ext{queries where gold cluster is selected}}
{\#	ext{queries}}
\]

This identifies whether failures originate in cluster routing or downstream retrieval.

---

## 18. Baseline retrieval preservation

Run:

```text
Baseline Verbatim-RAG
```

and save its top-K results.

Then run:

```text
Clustered Verbatim-RAG
```

Calculate:

\[
Overlap@K =
rac{|R_{baseline}@K \cap R_{clustered}@K|}
{|R_{baseline}@K|}
\]

This measures how much of the original Verbatim-RAG retrieval behavior is preserved.

---

## 19. Distinguish two types of recall

Report both:

### Ground-truth recall

```text
Does the system retrieve the gold evidence?
```

### Baseline preservation

```text
Does the system preserve the original Verbatim-RAG results?
```

These are not equivalent.

A clustered system may retrieve the correct evidence while changing some of the original top-K results.

---

## 20. Efficiency metrics

For every query measure:

### Candidate count

```python
num_candidates
```

### Candidate fraction

\[
CandidateFraction =
rac{N_{candidate}}
{N_{total}}
\]

### Search-space reduction

\[
Reduction =
1-CandidateFraction
\]

Example:

```text
Total chunks:       100,000
Candidates:          12,000

Candidate fraction: 12%
Reduction:           88%
```

---

## 21. Latency evaluation

Measure separately:

```text
embedding time
cluster routing time
candidate construction time
Verbatim-RAG retrieval time
total time
```

Report:

```text
mean
median
p95
```

Do not report only mean latency.

---

## 22. Random routing baseline

If the clustered system searches 10% of the corpus, create:

```text
Random 10% candidate selection
```

Compare:

```text
Random 10%
        vs
K-Means 10%
        vs
Full corpus
```

This determines whether semantic clustering provides useful routing rather than merely reducing the search space.

---

## 23. Offline K-Means baseline

Compare:

```text
A. Full Verbatim-RAG

B. Offline K-Means + Verbatim-RAG

C. Online K-Means + Verbatim-RAG
```

This isolates the value of the online formulation.

---

## 24. Online-update experiment

After the static experiment works, evaluate actual online behavior.

Start with:

```text
Dataset A
 ↓
build clusters
 ↓
evaluate
```

Then introduce additional documents:

```text
Dataset A
 ↓
partial_fit(new documents)
 ↓
evaluate
```

Repeat:

```text
A
A + 10%
A + 20%
A + 30%
...
```

Measure:

- routing recall
- gold retrieval recall
- candidate fraction
- number of clusters
- centroid movement
- retrieval latency

---

## 25. Dynamic clustering experiment

Only after the fixed-cluster experiment is complete, enable:

```text
dynamic cluster creation
merge
split
```

Compare:

```text
Fixed K-Means
      vs
Online K-Means
      vs
Online + dynamic creation
      vs
Online + creation + merge
      vs
Online + creation + merge + split
```

Do not introduce all capabilities simultaneously at the beginning.

---

## 26. Recommended experiment matrix

| System | Cluster routing | Final retrieval |
|---|---|---|
| Baseline | None | Verbatim-RAG |
| Random | Random subset | Verbatim-RAG |
| Offline K-Means | Top-N clusters | Verbatim-RAG |
| Online K-Means | Top-N clusters | Verbatim-RAG |

For each system measure:

```text
Gold Hit@1
Gold Hit@5
Gold Hit@10
Routing Recall
Baseline overlap@K
Candidate fraction
Candidate reduction
Mean latency
Median latency
P95 latency
```

---

## 27. Main plots

Generate at least:

### Plot 1 — Retrieval quality vs. candidate fraction

```text
x = percentage of corpus searched
y = Gold Recall@K
```

### Plot 2 — Latency vs. candidate fraction

```text
x = candidate fraction
y = retrieval latency
```

### Plot 3 — Routing recall vs. number of clusters searched

```text
x = top-N clusters
y = cluster routing recall
```

### Plot 4 — Number of clusters over time

```text
x = number of processed documents
y = number of active clusters
```

### Plot 5 — Baseline preservation

```text
x = candidate fraction
y = overlap@K with original Verbatim-RAG
```

---

## 28. Evaluation output format

Every experiment should save machine-readable results:

```text
results/
    baseline/
        predictions.jsonl
        metrics.json

    online_kmeans/
        k500_top5/
            predictions.jsonl
            metrics.json
        k500_top10/
            predictions.jsonl
            metrics.json
        k1000_top10/
            predictions.jsonl
            metrics.json

    random/
        10pct/
            predictions.jsonl
            metrics.json
```

A prediction record could contain:

```json
{
    "query_id": "...",
    "question": "...",
    "gold_paper": "...",
    "gold_chunk": 123,
    "selected_clusters": [17, 42, 81],
    "num_candidates": 8421,
    "candidate_fraction": 0.084,
    "retrieved_chunks": [...],
    "gold_hit_at_1": false,
    "gold_hit_at_5": true,
    "gold_hit_at_10": true,
    "routing_hit": true,
    "baseline_overlap_at_10": 0.8,
    "latency_ms": 53.2
}
```

---

## 29. Repository structure

Keep the implementation modular:

```text
verbatim-rag-onlineclustering/
│
├── src/
│   ├── clustering/
│   │   ├── online_kmeans.py
│   │   └── cluster_index.py
│   │
│   ├── retrieval/
│   │   ├── baseline.py
│   │   └── clustered.py
│   │
│   ├── datasets/
│   │   └── acl_verbatim.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── routing_metrics.py
│   │   ├── efficiency_metrics.py
│   │   └── evaluator.py
│   │
│   └── experiments/
│       ├── run_baseline.py
│       ├── run_clustered.py
│       ├── run_random_baseline.py
│       └── run_online_update.py
│
├── configs/
│   ├── baseline.yaml
│   ├── online_kmeans.yaml
│   └── experiments.yaml
│
├── scripts/
│   ├── download_acl.py
│   ├── build_embeddings.py
│   ├── build_cluster_index.py
│   └── run_evaluation.py
│
├── results/
│
└── tests/
    ├── test_online_kmeans.py
    ├── test_cluster_index.py
    ├── test_routing.py
    └── test_evaluation.py
```

---

## 30. Reproducibility requirements

Every experiment must record:

```text
random seed
embedding model
embedding dimension
distance metric
number of clusters
max clusters
new-cluster threshold
merge threshold
split threshold
number of papers
number of chunks
number of clusters searched
retrieval K
dataset version/config
```

Save the configuration alongside the results.

Example:

```yaml
dataset:
  name: KRLabsOrg/acl-verbatim-spans
  config: canonical
  evaluation_split: test

corpus:
  source: KRLabsOrg/acl-anthology-md
  num_papers: 5000

embedding:
  model: <existing Verbatim-RAG embedding model>
  normalize: true

clustering:
  algorithm: online_kmeans
  metric: cosine
  n_clusters: 500
  max_clusters: 500
  dynamic_creation: false
  merging: false
  splitting: false
  random_state: 42

routing:
  top_clusters: 10

retrieval:
  top_k: 10
```

---

## 31. Important evaluation constraint

The **gold ACL test set should drive evaluation, not corpus construction**.

The test dataset provides the ground truth:

```text
question
paper_id
chunk_index
chunk
label
spans
gold_paper
gold_chunk
```

The actual retrieval experiment should construct/search the underlying ACL corpus.

Conceptually:

```text
ACL Anthology
       ↓
your retrieval index
       ↓
your Online K-Means
```

while:

```text
acl-verbatim-spans/test
       ↓
evaluation ground truth
```

This prevents accidentally evaluating on an already-retrieved candidate set rather than testing the new retrieval layer.

---

## 32. Expected research result

The desired research conclusion is not necessarily:

> "Online K-Means improves Verbatim-RAG."

The more appropriate hypothesis is:

> **Online K-Means can substantially reduce the retrieval search space while maintaining a high proportion of the original Verbatim-RAG retrieval quality.**

The central comparison should be:

```text
                     Full corpus
                         │
              ┌──────────┴──────────┐
              │                     │
          Baseline             Online K-Means
              │                     │
         100% search          10–20% search
              │                     │
         Recall = X           Recall = Y
              │                     │
         Latency = A           Latency = B
```

The primary research question becomes:

\[
\boxed{
\text{How much can the search space be reduced for a given acceptable loss in retrieval quality?}
}
\]

---

## 33. Implementation order

Implement the project in the following phases.

### Phase 1 — Existing system baseline

1. Load ACL-Verbatim dataset.
2. Load a small ACL corpus.
3. Run existing Verbatim-RAG retrieval.
4. Save predictions.
5. Implement evaluation.

### Phase 2 — Cluster index

6. Integrate existing `OnlineKMeans`.
7. Embed corpus.
8. Cluster embeddings.
9. Create `cluster → chunk` index.
10. Implement top-N cluster selection.

### Phase 3 — Clustered retrieval

11. Restrict Verbatim-RAG candidates using selected clusters.
12. Preserve the original Verbatim-RAG retrieval code.
13. Run clustered retrieval.
14. Compare against baseline.

### Phase 4 — Evaluation

15. Implement gold evidence Hit@K.
16. Implement routing Recall@N.
17. Implement baseline overlap@K.
18. Implement candidate reduction.
19. Implement latency measurements.
20. Generate plots.

### Phase 5 — Controls

21. Add random candidate-selection baseline.
22. Add offline K-Means baseline.
23. Compare all systems.

### Phase 6 — Online behavior

24. Add incremental `partial_fit`.
25. Add corpus-growth experiment.
26. Measure cluster stability and retrieval quality.

### Phase 7 — Dynamic clustering

27. Enable dynamic cluster creation.
28. Evaluate merging.
29. Evaluate splitting.
30. Perform ablation study.

### Phase 8 — Scale

31. Increase from 1–5k papers → 10–20k.
32. Eventually run the full ACL corpus.
33. Produce final quality/efficiency curves.

---

## 34. Copilot implementation instruction

Use the following as the main instruction when asking GitHub Copilot to implement the project:

> **Do not redesign Verbatim-RAG. Implement Online K-Means as a coarse candidate-routing layer before the existing Verbatim-RAG retrieval stage. The primary research objective is to measure the trade-off between retrieval quality and the fraction of the corpus searched. Preserve the original retrieval implementation as a baseline and make every experiment reproducible. Start with fixed Online K-Means without cluster creation, merging, or splitting; add those capabilities only after the basic clustered retrieval experiment is validated.**
>
> **Use `KRLabsOrg/acl-verbatim-spans` (`canonical` configuration) for evaluation and `KRLabsOrg/acl-anthology-md` as the underlying ACL corpus. Start with 1,000–5,000 papers for development, then scale to 10,000–20,000 papers, and finally evaluate on the full corpus once the implementation is stable. Keep the gold test split strictly held out from parameter tuning.**
>
> **The primary evaluation should measure gold evidence Hit@K, cluster-routing Recall@N, overlap with baseline Verbatim-RAG retrieval, candidate-set fraction/reduction, and latency. Include random candidate selection and offline K-Means as control/baseline systems. Generate reproducible machine-readable results and quality-vs-efficiency plots.**
>
> **Implement the system incrementally according to the phases described above. Do not implement dynamic splitting/merging until the basic fixed Online K-Means routing experiment has been validated.**

> **Configuration-driven experiments: Implement the system so that experiment type and all relevant parameters (dataset size, batch size, clustering method, number of clusters, routing parameters, dynamic clustering options, retrieval settings, etc.) can be changed through configuration files without modifying the source code. A single experiment runner should support all defined experiment types.**
