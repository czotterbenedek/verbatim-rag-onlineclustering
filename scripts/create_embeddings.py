#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_path, load_config
from src.data_pipeline import read_jsonl
from src.embeddings import create_embeddings


def main():
    parser = argparse.ArgumentParser(description="Create and persist chunk/query embeddings.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data, embedding = config["data"], config["embedding"]
    chunks = read_jsonl(config_path(config, data["chunks_path"]))
    questions = read_jsonl(config_path(config, data["evaluation_path"]))
    unique_questions = list(dict.fromkeys(row["question"] for row in questions))
    chunk_vectors = create_embeddings([chunk["text"] for chunk in chunks], embedding["model"], embedding["batch_size"], embedding["normalize"])
    query_vectors = create_embeddings(unique_questions, embedding["model"], embedding["batch_size"], embedding["normalize"])
    chunk_path = config_path(config, embedding["path"])
    query_path = config_path(config, embedding["query_path"])
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    query_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(chunk_path, chunk_vectors)
    np.save(query_path, query_vectors)
    query_path.with_suffix(".json").write_text(json.dumps(unique_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved chunk embeddings: {chunk_vectors.shape} -> {chunk_path}")
    print(f"Saved query embeddings: {query_vectors.shape} -> {query_path}")


if __name__ == "__main__":
    main()