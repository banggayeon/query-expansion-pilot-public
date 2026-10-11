"""Read-only structural audit; no retrieval, expansion, or corpus scoring.

Run: python3 -m qepilot.audit
Only title/text are exposed by retrieval_documents; provenance and labels stay out.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from .spans import Span, validate_spans
from .text import overlap, TOKENIZER_VERSION


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_documents(records):
    return [{"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]} for r in records]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root=ROOT):
    config = json.loads((root / "configs/pilot.json").read_text())
    corpus = read_jsonl(root / "data/corpus.jsonl")
    queries = read_jsonl(root / "data/queries.jsonl")
    families = read_jsonl(root / "data/families.jsonl")
    draft = read_jsonl(root / "data/qrels.draft.jsonl")
    reviews = read_jsonl(root / "data/relevance_review.jsonl")
    semantic = read_jsonl(root / "data/semantic_review.jsonl")
    contrast = read_jsonl(root / "data/contrast_sets.jsonl")
    sources = read_jsonl(root / "data/sources.jsonl")
    dictionary = read_jsonl(root / "data/dictionary.jsonl")
    docs, qs = {r["doc_id"]: r for r in corpus}, {q["query_id"]: q for q in queries}
    fs = {f["family_id"]: f for f in families}
    _require(len(docs) == len(corpus) == config["corpus_size"] == 60, "document count/IDs")
    _require(len(fs) == len(families) == config["families"] == 12, "family count/IDs")
    _require(len(qs) == len(queries) == 36, "query count/IDs")
    _require(all(d["provenance"] == "independent_synthetic" for d in corpus), "non-synthetic corpus")
    _require(all(q["provenance"] == "independent_synthetic" for q in queries), "non-synthetic queries")
    _require(not any(config[k] for k in ("allow_search_execution", "allow_llm_execution", "allow_reranker_execution")), "execution gate")
    lengths = [len(d["text"].split()) for d in corpus]
    _require(all(80 <= n <= 140 for n in lengths), "document length outside plan")
    _require(len({d["text"] for d in corpus}) == 60, "duplicate documents")
    source_ids = {s["id"] for s in sources}
    for q in queries:
        _require(q["family_id"] in fs, "unknown query family")
        _require(q["split"] == fs[q["family_id"]]["split"], "family split leakage")
        spans = [Span(**{k: s[k] for k in ("start", "end", "kind", "sense")}) for s in q["spans"]]
        validate_spans(q["text"], spans)
        _require(all(q["text"][s["start"]:s["end"]] == s["text"] for s in q["spans"]), "span text mismatch")
    lexical = []
    for f in families:
        _require(set(f["query_ids"]) == {"q0", "qINV", "qDIR"}, "missing variant")
        _require(set(f["document_ids"]) <= docs.keys() and len(f["document_ids"]) == 5, "family docs")
        _require(set(f["source_refs"]) <= source_ids, "missing source reference")
        for variant, qid in f["query_ids"].items():
            _require(qid in qs and qs[qid]["family_id"] == f["family_id"] and qs[qid]["variant"] == variant, "family query mapping")
        if f["dir_type"] == "lexical_confusable":
            lexical.append({"family_id": f["family_id"], "old_term": f["old_term"], "new_term": f["new_term"],
                            "whole_query": overlap(qs[f["query_ids"]["q0"]]["text"], qs[f["query_ids"]["qDIR"]]["text"]),
                            "term_only": overlap(f["old_term"], f["new_term"]),
                            "semantic_review_status": f["semantic_review_status"]})
    expected = {(qid, d) for f in families for qid in f["query_ids"].values() for d in f["document_ids"]}
    for name, rows in (("draft", draft), ("review", reviews)):
        _require(len(rows) == len(expected) and {(r["query_id"], r["doc_id"]) for r in rows} == expected, name + " cells")
    _require(all(r["status"] == "draft" and not r["independently_reviewed"] for r in draft), "synthetic label promotion")
    _require(all(type(r["grade"]) is int and r["grade"] in (0, 1, 2) for r in draft), "draft grade")
    _require(all(r["status"] == "unjudged" and r["grade"] is None and not r["reviews"] for r in reviews), "unexpected human review")
    _require(len(semantic) == 24 and all(r["status"] == "unjudged" for r in semantic), "semantic review tasks")
    _require(len(contrast) == len(lexical) == 4, "lexical contrast count")
    for c in contrast:
        _require(set(c["roles"]) == {"original_relevant", "changed_relevant", "hard_negative"}, "contrast roles")
        ids = set(c["roles"].values())
        _require(len(ids) == 3 and ids <= docs.keys(), "contrast document IDs")
        _require(c["query_ids"] == fs[c["family_id"]]["query_ids"], "contrast variants")
        _require(all((q, d) in expected for q in c["query_ids"].values() for d in ids), "missing contrast review cell")
    _require(all(d["source_id"] in source_ids and not d["auto_activate"] for d in dictionary), "dictionary provenance/activation")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("data", "configs") for p in sorted((root / folder).glob("*")) if p.is_file()}
    return {"audit_status": "structure_valid", "scope": "data_schema_only_no_search",
            "documents": 60, "queries": 36, "families": 12,
            "splits": dict(Counter(f["split"] for f in families)),
            "dir_types": dict(Counter(f["dir_type"] for f in families)),
            "word_count": {"min": min(lengths), "max": max(lengths)},
            "draft_judgment_cells": len(draft), "human_reviewed_cells": 0,
            "contrast_sets": 4, "contrast_required_cells": 36, "contrast_reviewed_cells": 0,
            "contrast_discrimination": "not_run; independent relevance and rankings pending",
            "qrels_status": "draft", "cpu_search_executed": False, "llm_expansion_executed": False,
            "tokenizer": TOKENIZER_VERSION, "lexical_overlap": lexical, "sha256": hashes}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
