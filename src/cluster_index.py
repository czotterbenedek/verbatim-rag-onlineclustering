import time

import numpy as np

from .OnlineKMeans import OnlineKMeans


class ClusteredRetrievalIndex:
    def __init__(self, embeddings, n_clusters, metric="cosine", random_state=42,
                 batch_size=1024, algorithm="online_kmeans", max_clusters=None,
                 new_cluster_threshold=None, merge_threshold=None,
                 split_conductance_threshold=None):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.algorithm = algorithm
        self.metric = metric
        if algorithm == "online_kmeans":
            self.kmeans = OnlineKMeans(
                n_clusters=n_clusters,
                max_clusters=max_clusters or n_clusters,
                metric=metric,
                random_state=random_state,
                new_cluster_threshold=new_cluster_threshold,
                merge_threshold=merge_threshold,
                split_conductance_threshold=split_conductance_threshold,
            )
            for start in range(0, len(self.embeddings), batch_size):
                self.kmeans.partial_fit(self.embeddings[start:start + batch_size])
        elif algorithm == "offline_kmeans":
            from sklearn.cluster import KMeans

            self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            self.kmeans.fit(self.embeddings)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        self._refresh_assignments()

    def _refresh_assignments(self):
        self.chunk_to_cluster = self.kmeans.predict(self.embeddings)
        centroid_count = (len(self.kmeans.centroids) if self.algorithm == "online_kmeans"
                          else len(self.kmeans.cluster_centers_))
        self.cluster_to_chunks = {
            cluster: np.flatnonzero(self.chunk_to_cluster == cluster)
            for cluster in range(centroid_count)
        }

    def append_embeddings(self, embeddings, batch_size=1024):
        """Add new embeddings and refresh assignments for the visible corpus."""
        if self.algorithm != "online_kmeans":
            raise ValueError("Only online_kmeans indexes support incremental updates.")
        new_embeddings = np.asarray(embeddings, dtype=np.float32)
        if new_embeddings.ndim != 2 or len(new_embeddings) == 0:
            raise ValueError("New embeddings must be a non-empty 2D array.")
        if self.embeddings.ndim == 2 and self.embeddings.shape[1] != new_embeddings.shape[1]:
            raise ValueError("New embeddings have a different dimension from the index.")

        previous_centroids = None if self.kmeans.centroids is None else self.kmeans.centroids.copy()
        update_started = time.perf_counter()
        for start in range(0, len(new_embeddings), batch_size):
            self.kmeans.partial_fit(new_embeddings[start:start + batch_size])
        update_latency_ms = (time.perf_counter() - update_started) * 1000

        assignment_started = time.perf_counter()
        self.embeddings = np.concatenate([self.embeddings, new_embeddings], axis=0)
        self._refresh_assignments()
        assignment_latency_ms = (time.perf_counter() - assignment_started) * 1000

        centroid_drift = 0.0
        if previous_centroids is not None:
            shared = min(len(previous_centroids), len(self.kmeans.centroids))
            if shared:
                centroid_drift = float(np.mean(np.linalg.norm(
                    self.kmeans.centroids[:shared] - previous_centroids[:shared], axis=1
                )))
        return {
            "update_latency_ms": update_latency_ms,
            "assignment_latency_ms": assignment_latency_ms,
            "total_latency_ms": update_latency_ms + assignment_latency_ms,
            "throughput_chunks_per_second": len(new_embeddings) / max(update_latency_ms / 1000, 1e-12),
            "centroid_drift_mean": centroid_drift,
            "num_clusters": int(len(self.kmeans.centroids)),
            "total_seen": int(self.kmeans.total_seen),
        }

    def predict_top_clusters(self, query_embedding, n_clusters):
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self.algorithm == "online_kmeans":
            order, distances = self.kmeans.predict_top_clusters(query, n_clusters)
            return order[0].astype(int), distances[0]
        if self.metric == "cosine":
            query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
            centroids = self.kmeans.cluster_centers_ / np.maximum(
                np.linalg.norm(self.kmeans.cluster_centers_, axis=1, keepdims=True), 1e-12
            )
            distances = 1.0 - query @ centroids.T
        else:
            distances = np.linalg.norm(query[:, None, :] - self.kmeans.cluster_centers_[None, :, :], axis=2)
        order = np.argsort(distances[0], kind="stable")[:min(n_clusters, len(distances[0]))]
        return order.astype(int), distances[0, order]

    def candidate_ids_from_clusters(self, clusters):
        if len(clusters) == 0:
            return np.array([], dtype=int)
        return np.unique(np.concatenate([self.cluster_to_chunks[int(cluster)] for cluster in clusters]))

    def candidate_ids(self, query_embedding, n_clusters):
        clusters, distances = self.predict_top_clusters(query_embedding, n_clusters)
        return self.candidate_ids_from_clusters(clusters), clusters, distances