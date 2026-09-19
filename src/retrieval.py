import json

import numpy as np


class _QueryEmbeddingProvider:
    """Embedding provider that returns persisted query vectors."""

    def __init__(self, query_embeddings):
        self.query_embeddings = query_embeddings

    def embed_text(self, text):
        try:
            return self.query_embeddings[text].tolist()
        except KeyError as error:
            raise KeyError(f"No persisted embedding found for query: {text}") from error


class _InMemoryVectorStore:
    """VectorStore used by VerbatimIndex for reproducible local experiments."""

    enable_full_text = False

    def __init__(self, chunks, embeddings):
        from verbatim_rag.vector_stores.base import SearchResult

        self._search_result = SearchResult
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.by_id = {chunk["chunk_id"]: position for position, chunk in enumerate(chunks)}

    def add_vectors(self, ids, dense_vectors, sparse_vectors, texts, enhanced_texts, metadatas):
        raise NotImplementedError("This experiment store uses persisted embeddings.")

    def delete(self, ids):
        raise NotImplementedError("Deletion is not part of static evaluation.")

    def query(self, dense_query=None, sparse_query=None, text_query=None, top_k=5,
              search_type="dense", filter=None, **kwargs):
        candidate_ids = None if not filter else json.loads(filter)["candidate_ids"]
        positions = (np.arange(len(self.chunks)) if candidate_ids is None else
                     np.asarray([self.by_id[item] for item in candidate_ids], dtype=int))
        if len(positions) == 0:
            return []
        query = np.asarray(dense_query, dtype=np.float32)
        query /= max(np.linalg.norm(query), 1e-12)
        vectors = self.embeddings[positions]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        scores = vectors @ query
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [self._search_result(
            id=self.chunks[positions[position]]["chunk_id"],
            score=float(scores[position]),
            metadata={"chunk_id": self.chunks[positions[position]]["chunk_id"],
                      "paper_id": self.chunks[positions[position]]["paper_id"],
                      "chunk_index": self.chunks[positions[position]]["chunk_index"]},
            text=self.chunks[positions[position]]["text"],
        ) for position in order]


class VerbatimRAGRetriever:
    """Final retrieval through Verbatim-RAG's VerbatimIndex query path."""

    def __init__(self, chunks, embeddings, query_embeddings):
        from verbatim_rag import VerbatimIndex

        self.store = _InMemoryVectorStore(chunks, embeddings)
        self.index = VerbatimIndex(
            vector_store=self.store,
            dense_provider=_QueryEmbeddingProvider(query_embeddings),
        )

    def search(self, question, top_k=10, candidate_ids=None):
        filter_expression = None
        if candidate_ids is not None:
            filter_expression = json.dumps({
                "candidate_ids": [self.store.chunks[int(position)]["chunk_id"]
                                  for position in candidate_ids]
            })
        results = self.index.query(
            text=question,
            k=top_k,
            search_type="dense",
            filter=filter_expression,
        )
        return [result.id for result in results], [float(result.score) for result in results]
