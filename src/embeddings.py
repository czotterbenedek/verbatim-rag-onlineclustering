import numpy as np


def create_embeddings(texts, model_name, batch_size=32, normalize=True, device=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return np.asarray(model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=normalize), dtype=np.float32)