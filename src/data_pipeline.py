import json
from pathlib import Path


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = max(1, chunk_size - overlap)
    return [" ".join(words[start:start + chunk_size])
            for start in range(0, len(words), step)
            if words[start:start + chunk_size]]


def load_local_corpus(root, max_documents=None):
    paths = sorted(Path(root).rglob("*.md")) + sorted(Path(root).rglob("*.markdown"))
    selected = paths[:max_documents] if max_documents else paths
    return [{"paper_id": path.stem, "text": path.read_text(encoding="utf-8")} for path in selected]


def load_huggingface_corpus(dataset_name, split="train", max_documents=None,
                            config_name=None, required_paper_ids=None):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=bool(max_documents))
    records = []
    required_paper_ids = {str(paper_id) for paper_id in (required_paper_ids or set())}
    selected_ids = set()
    for row in dataset:
        text = row.get("markdown") or row.get("text") or row.get("content") or ""
        paper_id = (row.get("anthology_id") or row.get("paper_id") or
                    row.get("id") or row.get("doc_id"))
        if text and paper_id is not None and (not max_documents or
                                              len(records) < max_documents or
                                              str(paper_id) in required_paper_ids):
            records.append({"paper_id": str(paper_id), "text": str(text)})
            selected_ids.add(str(paper_id))
        if max_documents and len(records) >= max_documents and required_paper_ids.issubset(selected_ids):
            break
    return records


def build_chunks(documents, chunk_size=256, overlap=32, strategy="verbatim_rag",
                 min_chunk_size=500, max_chunk_size=5000):
    if strategy == "verbatim_rag":
        from verbatim_rag.chunker_providers import MarkdownChunkerProvider

        chunker = MarkdownChunkerProvider(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    chunks = []
    for document in documents:
        if strategy == "verbatim_rag":
            chunk_texts = [enhanced_text for _, enhanced_text in chunker.chunk(document["text"])]
        elif strategy == "words":
            chunk_texts = _chunk_text(document["text"], chunk_size, overlap)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        for index, text in enumerate(chunk_texts):
            chunks.append({"chunk_id": f"{document['paper_id']}_{index}", "paper_id": document["paper_id"], "chunk_index": index, "text": text})
    return chunks


def load_evaluation_rows(dataset_name, config_name, split):
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(dataset_name, config_name, split=split)]