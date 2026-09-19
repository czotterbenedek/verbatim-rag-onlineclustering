#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_path, load_config
from src.data_pipeline import (build_chunks, load_evaluation_rows, load_huggingface_corpus,
                               load_local_corpus, write_jsonl)


def main():
    parser = argparse.ArgumentParser(description="Prepare corpus chunks and evaluation rows.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    rows = load_evaluation_rows(data["evaluation_name"], data["evaluation_config"], data["evaluation_split"])
    required_paper_ids = {str(row["gold_paper"]) for row in rows if row.get("gold_paper")}
    include_evaluation_papers = data.get("include_evaluation_papers", False)
    if data["evaluation_split"] == "test" and include_evaluation_papers:
        raise ValueError("Do not build the held-out test corpus from test gold papers.")
    if data["corpus_source"] == "local":
        documents = load_local_corpus(config_path(config, data["local_corpus_dir"]), data.get("max_documents"))
    else:
        documents = load_huggingface_corpus(
            data["corpus_name"], data["corpus_split"], data.get("max_documents"),
            data.get("corpus_config", "fulltext"),
            required_paper_ids if include_evaluation_papers else None,
        )
    chunks = build_chunks(documents, config["chunking"].get("size"),
                          config["chunking"].get("overlap", 0),
                          config["chunking"].get("strategy", "verbatim_rag"),
                          config["chunking"].get("min_size", 500),
                          config["chunking"].get("max_size", 5000))
    write_jsonl(config_path(config, data["chunks_path"]), chunks)
    write_jsonl(config_path(config, data["evaluation_path"]), rows)
    indexed_ids = {chunk["paper_id"] for chunk in chunks}
    covered_ids = indexed_ids.intersection(required_paper_ids)
    print(f"Gold paper coverage: {len(covered_ids)}/{len(required_paper_ids)}")
    if required_paper_ids and not covered_ids:
        raise RuntimeError("No evaluation gold papers are present in the prepared corpus.")
    print(f"Prepared {len(documents)} documents and {len(chunks)} chunks.")
    print(f"Saved evaluation split '{data['evaluation_split']}' with {len(rows)} rows.")


if __name__ == "__main__":
    main()