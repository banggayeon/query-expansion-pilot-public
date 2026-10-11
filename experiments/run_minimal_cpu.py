"""Approved five-condition CPU run; frozen qepilot modules are imported, not edited.

All output is generated run data. No LLM, ES, GPU, reranker, or source-data edits.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"

from qepilot.audit import read_jsonl, retrieval_documents
from qepilot.evaluation import Qrels, evaluate, evaluate_dir, evaluate_inv, contrast_family
from qepilot.scoring import ScoringConfig, bm25, phrase_tf, synonym_or
from qepilot.spans import Span, Alias, query_groups, accept_aliases
from qepilot.statistics import paired_summary
from qepilot.text import tokens

RUN_ID = "cpu_minimal_20261011"
METHODS = ("bm25_O", "bm25_D", "dense_O", "hybrid_OO", "hybrid_DO")
KS = (10, 20, 50, 100)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def current_rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def timed(name, function, timing):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    timing[name] = {"wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                    "rss_mib_at_end": current_rss_mib(),
                    "process_peak_rss_mib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    return value


def rank_scores(scores):
    return sorted(scores, key=lambda d: (-scores[d], d))


def rrf(left, right):
    result = defaultdict(float)
    for ranks in (left, right):
        for i, doc_id in enumerate(ranks, 1):
            result[doc_id] += 1 / (60 + i)
    return dict(result)


def make_expansions(queries, dictionary, sources):
    groups_o, groups_d, logs = {}, {}, []
    for q in queries:
        spans = [Span(**{k: s[k] for k in ("start", "end", "kind", "sense")}) for s in q["spans"]]
        proposals = []
        for s in spans:
            if s.kind != "alias_expandable_semantic":
                continue
            text = q["text"][s.start:s.end]
            for entry in dictionary:
                if entry["sense"] != s.sense or entry["status"] != "source_checked":
                    continue
                forms = (entry["term"], entry["alias"])
                if tokens(text) not in [tokens(f) for f in forms]:
                    continue
                alias = forms[1] if tokens(text) == tokens(forms[0]) else forms[0]
                proposals.append(Alias(s.start, s.end, alias, s.sense, True,
                                       sources[entry["source_id"]]["url"], "ai_source_qc_not_human"))
        accepted, rejected = accept_aliases(q["text"], spans, proposals)
        groups_o[q["query_id"]], _ = query_groups(q["text"], spans)
        groups_d[q["query_id"]], _ = query_groups(q["text"], spans, proposals)
        logs.append({"query_id": q["query_id"], "family_id": q["family_id"], "variant": q["variant"],
                     "raw_query": q["text"], "expanded": bool(accepted),
                     "accepted": [asdict(a) for a in accepted],
                     "rejected": [{"proposal": asdict(r["proposal"]), "reason": r["reason"]} for r in rejected],
                     "groups_O": groups_o[q["query_id"]], "groups_D": groups_d[q["query_id"]],
                     "provenance": "existing four source-checked entries, explicitly activated for authorized provisional run",
                     "reviewer_kind": "ai", "human_reviewed": False})
    return groups_o, groups_d, logs


def term_statistics(documents, group_sets):
    docs = {d["doc_id"]: tokens(d["title"] + " " + d["text"]) for d in documents}
    terms = {tokens(t) for groups in group_sets for gs in groups.values() for original, aliases in gs for t in (original, *aliases)}
    freqs, dfs = {}, {}
    for term in terms:
        freqs[term] = {d: phrase_tf(ts, " ".join(term)) for d, ts in docs.items()}
        dfs[term] = sum(v > 0 for v in freqs[term].values())
    return docs, freqs, dfs


def lexical_scores(groups, index):
    docs, freqs, dfs = index
    avg = statistics.mean(len(ts) for ts in docs.values())
    out = {}
    for qid, gs in groups.items():
        scores = {}
        for doc_id, ts in docs.items():
            def base(term):
                key = tokens(term)
                return bm25(freqs[key][doc_id], dfs[key], len(docs), len(ts), avg)
            scores[doc_id] = synonym_or([(base(o), [base(a) for a in aliases]) for o, aliases in gs])
        out[qid] = scores
    return out


def load_dense(manifest):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    pooling = json.loads(Path(manifest["files"]["1_Pooling/config.json"]).read_text())
    assert pooling["pooling_mode_mean_tokens"] and not pooling["pooling_mode_cls_token"] and not pooling["pooling_mode_max_tokens"]
    sentence_cfg = json.loads(Path(manifest["files"]["sentence_bert_config.json"]).read_text())
    assert sentence_cfg["max_seq_length"] == 256
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(manifest["files"]["onnx/model.onnx"], sess_options=options,
                                   providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    tokenizer = Tokenizer.from_file(manifest["files"]["tokenizer.json"])
    return session, tokenizer


def encode_dense(texts, session, tokenizer):
    import numpy as np
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = [len(e.ids) for e in tokenizer.encode_batch(texts)]
    if any(n > 256 for n in lengths):
        raise RuntimeError("Truncation would occur; stop instead of hiding lost constraints")
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    vectors = []
    required = {i.name for i in session.get_inputs()}
    for offset in range(0, len(texts), 16):
        enc = tokenizer.encode_batch(texts[offset:offset + 16])
        feed = {"input_ids": np.asarray([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in enc], dtype=np.int64)}
        output = session.run(None, {k: v for k, v in feed.items() if k in required})[0]
        assert output.ndim == 3 and output.shape[-1] == 384 and output.dtype == np.float32
        mask = feed["attention_mask"].astype(np.float32)[..., None]
        pooled = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), np.float32(1e-9))
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise RuntimeError("zero embedding")
        vectors.append(pooled / norm)
    return np.concatenate(vectors), {"max_tokens": max(lengths), "truncated_count": 0, "lengths": lengths}


def validate_qc(root, queries, documents, draft):
    qcroot = root / "qc" / RUN_ID
    qc = read_jsonl(qcroot / "relevance.ai.jsonl")
    semantic = read_jsonl(qcroot / "semantic.ai.jsonl")
    docs = {d["doc_id"]: d for d in documents}
    contrasts = read_jsonl(root / "data/contrast_sets.jsonl")
    expected = {(q, d) for c in contrasts for q in c["query_ids"].values() for d in c["roles"].values()}
    assert len(qc) == len(expected) == 36
    assert {(r["query_id"], r["doc_id"]) for r in qc} == expected
    for r in qc:
        assert r["reviewer_kind"] == "ai" and not r["human_reviewed"]
        assert r["qc_grade"] in (0, 1, 2) and r["evidence"] in docs[r["doc_id"]]["text"]
    assert len(semantic) == 24
    draft_map = {(r["query_id"], r["doc_id"]): r["grade"] for r in draft}
    disagreements = [r for r in qc if r["qc_grade"] != draft_map[r["query_id"], r["doc_id"]]]
    return qc, {"relevance_cells": 36, "semantic_pairs": 24, "human_reviews": 0,
                "reviewer_kind": "ai", "author_independent": False,
                "grade_disagreements_vs_draft": len(disagreements),
                "ambiguous_cells": sum(r["ambiguity"] for r in qc),
                "semantic_verdicts": dict(Counter(r["verdict"] for r in semantic)),
                "data_modified": False}


def grouped_metrics(records, queries):
    meta = {q["query_id"]: q for q in queries}
    out = []
    for layer in sorted({r["label_layer"] for r in records}):
        for method in METHODS:
            for k in KS:
                selected = [r for r in records if r["label_layer"] == layer and r["method"] == method and r["k"] == k and r["raw"]["status"] == "ok"]
                if not selected:
                    continue
                groups = {"all": selected}
                for field in ("split", "variant", "dir_type", "domain", "inv_type"):
                    for value in sorted({meta[r["query_id"]][field] for r in selected}):
                        groups[f"{field}={value}"] = [r for r in selected if meta[r["query_id"]][field] == value]
                for group, rows in groups.items():
                    stats = {}
                    for metric in ("recall", "ndcg", "mrr", "precision_known", "success", "success_known", "judged_coverage", "unjudged"):
                        by_family = defaultdict(list)
                        for r in rows:
                            value = r["raw"][metric]
                            if value is not None:
                                by_family[meta[r["query_id"]]["family_id"]].append(value)
                        means = [statistics.mean(v) for v in by_family.values()]
                        stats[metric] = statistics.mean(means) if means else None
                        stats[metric + "_valid_queries"] = sum(len(v) for v in by_family.values())
                    out.append({"label_layer": layer, "method": method, "k": k, "group": group,
                                "n_queries": len(rows), "n_families": len({meta[r["query_id"]]["family_id"] for r in rows}),
                                "warning": "provisional; unknown judgments not certified irrelevant", **stats})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing run")
    baseline = json.loads((ROOT / "reports/data_audit.json").read_text())["sha256"]
    for relative, sha in baseline.items():
        if digest(ROOT / relative) != sha:
            raise RuntimeError(f"Frozen input changed: {relative}")
    frozen_code = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "qepilot").glob("*.py")}
    policy = json.loads((ROOT / "configs" / (RUN_ID + ".json")).read_text())
    assert tuple(policy["allowed_methods"]) == METHODS and policy["cpu_only"]
    assert not any(policy[k] for k in ("llm_expansion", "reranker", "es"))
    model_manifest = json.loads(Path(args.model_manifest).read_text())
    assert model_manifest["model_id"] == policy["model_id"]
    full_documents = read_jsonl(ROOT / "data/corpus.jsonl")
    documents = retrieval_documents(full_documents)
    queries = read_jsonl(ROOT / "data/queries.jsonl")
    families = read_jsonl(ROOT / "data/families.jsonl")
    draft = read_jsonl(ROOT / "data/qrels.draft.jsonl")
    qc, qc_summary = validate_qc(ROOT, queries, documents, draft)
    dictionary = read_jsonl(ROOT / "data/dictionary.jsonl")
    sources = {s["id"]: s for s in read_jsonl(ROOT / "data/sources.jsonl")}
    assert {d["term"] for d in dictionary} == set(policy["dictionary_terms"])
    output.mkdir(parents=True)
    timings = {}
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    go, gd, expansions = timed("dictionary_and_query_groups", lambda: make_expansions(queries, dictionary, sources), timings)
    index = timed("lexical_shared_index", lambda: term_statistics(documents, [go, gd]), timings)
    scores = {}
    scores["bm25_O"] = timed("bm25_O_score", lambda: lexical_scores(go, index), timings)
    scores["bm25_D"] = timed("bm25_D_score", lambda: lexical_scores(gd, index), timings)
    session, tokenizer = timed("dense_import_and_model_load", lambda: load_dense(model_manifest), timings)
    dv, dlengths = timed("dense_document_encode", lambda: encode_dense([d["title"] + " " + d["text"] for d in documents], session, tokenizer), timings)
    qv, qlengths = timed("dense_query_encode", lambda: encode_dense([q["text"] for q in queries], session, tokenizer), timings)
    similarities = timed("dense_cosine_score", lambda: qv @ dv.T, timings)
    import numpy as np
    np.save(output / "document_embeddings.npy", dv)
    np.save(output / "query_embeddings.npy", qv)
    scores["dense_O"] = {q["query_id"]: {d["doc_id"]: float(similarities[i, j]) for j, d in enumerate(documents)} for i, q in enumerate(queries)}
    rankings = {m: {qid: rank_scores(s) for qid, s in per_query.items()} for m, per_query in scores.items()}
    for method, lexical in (("hybrid_OO", "bm25_O"), ("hybrid_DO", "bm25_D")):
        scores[method] = timed(method + "_fusion", lambda lexical=lexical: {q["query_id"]: rrf(rankings[lexical][q["query_id"]], rankings["dense_O"][q["query_id"]]) for q in queries}, timings)
        rankings[method] = {qid: rank_scores(s) for qid, s in scores[method].items()}
    search_wall, search_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    corpus_ids = frozenset(d["doc_id"] for d in documents)
    layers = {}
    for name, rows, grade_key in (("ai_qc_36", qc, "qc_grade"), ("author_draft_180", draft, "grade")):
        mapping = defaultdict(dict)
        for r in rows:
            mapping[r["query_id"]][r["doc_id"]] = r[grade_key]
        layers[name] = {qid: Qrels(gs, corpus_ids, "draft", RUN_ID + "_" + name) for qid, gs in mapping.items()}
    metrics, dirs, invs, paired = [], [], [], []
    for layer, qr in layers.items():
        keyed = {}
        for method in METHODS:
            for qid, rel in qr.items():
                for k in KS:
                    raw = evaluate(rankings[method][qid], rel, k, allow_draft=True)
                    keyed[method, qid, k] = raw
                    metrics.append({"label_layer": layer, "method": method, "query_id": qid, "k": k,
                                    "run_interpretation": "provisional_AI_QC" if layer == "ai_qc_36" else "unvalidated_author_draft_diagnostic", "raw": raw})
            for f in families:
                ids = f["query_ids"]
                if not set(ids.values()) <= qr.keys():
                    continue
                dirs.append({"label_layer": layer, "method": method, "family_id": f["family_id"],
                             "raw": evaluate_dir(rankings[method][ids["q0"]], rankings[method][ids["qDIR"]], qr[ids["q0"]], qr[ids["qDIR"]], k=10, allow_draft=True)})
                for k in (10, 20, 50):
                    invs.append({"label_layer": layer, "method": method, "family_id": f["family_id"], "k": k,
                                 "raw": evaluate_inv(keyed[method, ids["q0"], k], keyed[method, ids["qINV"], k])})
        for before, after in (("bm25_O", "bm25_D"), ("hybrid_OO", "hybrid_DO")):
            for k in (10, 20, 50):
                rows = [{"family_id": q["family_id"], "variant": q["variant"],
                         "before": keyed[before, q["query_id"], k], "after": keyed[after, q["query_id"], k],
                         "snapshot_id": RUN_ID + "_" + layer, "comparison": after + "-" + before}
                        for q in queries if q["query_id"] in qr]
                for metric in ("recall", "ndcg"):
                    paired.append({"label_layer": layer, "comparison": after + "-" + before, "k": k, "metric": metric,
                                   "summary": paired_summary(rows, metric=metric)})
    contrasts = []
    for c in read_jsonl(ROOT / "data/contrast_sets.jsonl"):
        for method in METHODS:
            result = contrast_family(scores[method], layers["ai_qc_36"], c["query_ids"], c["roles"], allow_draft=True)
            contrasts.append({"family_id": c["family_id"], "method": method, "reviewer_kind": "ai", "human_reviewed": False,
                              "ambiguity_flag": c["family_id"] == "f04", "raw": result})
    ranking_rows, topk, changes = [], [], []
    for method in METHODS:
        for q in queries:
            qid = q["query_id"]
            rank = rankings[method][qid]
            assert len(rank) == len(set(rank)) == 60 and set(rank) == corpus_ids
            ranking_rows.append({"method": method, "query_id": qid, "ranked_docs": [{"doc_id": d, "rank": i, "score": scores[method][qid][d]} for i, d in enumerate(rank, 1)]})
            for k in KS:
                topk.append({"method": method, "query_id": qid, "k": k, "status": "N/A" if k > 60 else "ok", "doc_ids": None if k > 60 else rank[:k]})
    for before, after in (("bm25_O", "bm25_D"), ("hybrid_OO", "hybrid_DO")):
        for q in queries:
            qid = q["query_id"]
            a, b = rankings[before][qid], rankings[after][qid]
            ra, rb = {d: i for i, d in enumerate(a, 1)}, {d: i for i, d in enumerate(b, 1)}
            for k in (10, 20, 50):
                changes.append({"comparison": after + "-" + before, "query_id": qid, "k": k,
                                "topk_overlap_count": len(set(a[:k]) & set(b[:k])),
                                "entered": sorted(set(b[:k]) - set(a[:k])), "exited": sorted(set(a[:k]) - set(b[:k])),
                                "same_order": a[:k] == b[:k],
                                "rank_shifts": [{"doc_id": d, "before": ra[d], "after": rb[d], "improvement": ra[d] - rb[d]} for d in a if ra[d] != rb[d]]})
    for relative, sha in baseline.items():
        assert digest(ROOT / relative) == sha, relative
    for relative, sha in frozen_code.items():
        assert digest(ROOT / relative) == sha, relative
    manifest = {"run_id": RUN_ID, "status": "provisional", "human_reviewed": False, "methods": METHODS,
                "documents": 60, "queries": 36, "ks": KS, "K100": "N/A", "threads": 4,
                "model": {**model_manifest, "file_sha256": {k: digest(Path(v)) for k, v in model_manifest["files"].items()}},
                "backend": "onnxruntime CPU float32, masked mean pooling, L2 normalization", "scoring": ScoringConfig().manifest(),
                "data_and_original_config_sha256": baseline, "frozen_code_sha256": frozen_code,
                "source_unchanged": True, "qc_input_sha256": {p.name: digest(p) for p in (ROOT / "qc" / RUN_ID).glob("*")},
                "runner_sha256": digest(Path(__file__)), "policy": policy,
                "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                                "cpu_model": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), "unknown"),
                                "packages": {p: importlib.metadata.version(p) for p in ("onnxruntime", "numpy", "tokenizers", "huggingface-hub")}},
                "timings": timings, "search_pipeline_wall_seconds": search_wall, "search_pipeline_cpu_seconds": search_cpu,
                "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "timing_note": "one pass, shared index/embeddings cached; phase peaks are cumulative, not standalone method peaks; excludes dependency/model download and evaluation",
                "token_lengths": {"documents": dlengths, "queries": qlengths},
                "embedding_order": {"documents": [d["doc_id"] for d in documents], "queries": [q["query_id"] for q in queries]},
                "qc": qc_summary,
                "expansion_applied": sum(e["expanded"] for e in expansions), "expansion_not_applied": sum(not e["expanded"] for e in expansions)}
    for filename, rows in (("expansions.jsonl", expansions), ("rankings.jsonl", ranking_rows), ("topk.jsonl", topk),
                           ("metrics.jsonl", metrics), ("dir.jsonl", dirs), ("inv.jsonl", invs),
                           ("paired.jsonl", paired), ("contrast.jsonl", contrasts), ("rank_changes.jsonl", changes),
                           ("grouped_metrics.jsonl", grouped_metrics(metrics, queries))):
        write_jsonl(output / filename, rows)
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(output), "methods": METHODS, "qc": qc_summary,
                      "expansion_applied": manifest["expansion_applied"], "search_wall_seconds": search_wall,
                      "cpu_seconds": search_cpu, "peak_rss_mib": manifest["process_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
