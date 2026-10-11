# Query Expansion Pilot — Handoff (Draft)

Updated: 2026-10-11. **Review before sharing or publishing.** This is a working handoff based on an independent synthetic retrieval pilot, not a verified reproduction of past experiments. The public repository is a curated subset; original reports and saved ranking artifacts are not included.

## Repository and dataset

- 60 synthetic documents, 12 families, 36 queries: `q0` (original intent), `qINV` (meaning-preserving rewrite), and `qDIR` (changed intent).
- `data/corpus.jsonl`, `data/queries.jsonl`, `data/qrels.draft.jsonl`: passages, queries, and **draft** judgments.
- `data/relevance_review.jsonl`, `data/semantic_review.jsonl`: review materials.
- `data/dictionary.jsonl`, `data/sources.jsonl`: limited alias dictionary and public terminology references.
- `qepilot/`: scoring, evaluation, audit and span-protection modules.
- `experiments/`, `configs/`, `tests/`: experiment scripts, configurations, and tests.

Basic checks:

```bash
python3 -m unittest discover -s tests -v
python3 -m qepilot.audit
```

Some scripts require extra packages or historical artifacts not included in the public repository. Running the checks above does not reproduce past ranking metrics.

## Earlier synthetic-pilot observations (provisional)

### CPU retrieval

Original and dictionary-expanded BM25, original dense retrieval, and hybrid conditions were compared. Dictionary expansion applied to a subset of queries. It helped recover certain relevant documents but sometimes also promoted hard negatives. Rank improvements alone do not establish semantic disambiguation.

### Focused BM25 ablation

A 2×2×2 design examined four lexical-confusable query families (12 queries) with three binary factors:

- **F:** suppress certain unprotected independent function-word groups.
- **N:** normalize specified number/unit expressions.
- **P:** use a token-sum-replacement phrase-scoring variant.

None of the configurations consistently corrected all failures. Function-word suppression could produce ties or regressions; number normalization sometimes improved targets and confusable negatives together; phrase-scoring changes improved some pairwise comparisons without resolving all errors. Interpret these as observations from a small synthetic diagnostic, not a general performance claim.

The original saved-run outputs and report appendices are **not included** in this handoff and must be retrieved from an authorized source before quoting exact historical scores as independently verified results.

## Suggested next steps

1. **Read-only reconnaissance:** inspect checkout, identify missing artifacts, inspect configs and dependencies, and run tests/audit. Record commit, hashes and environment.
2. **Independent relevance review:** blind-review query–passage relationships, with special attention to f04 and potential partial-relevance / shared-relevance cases. Preserve original draft labels until adjudication.
3. **Frozen-candidate reranker pilot:** compare scores and ranks on the **same** candidate set, tracking corrections, regressions and judged coverage. A reranker cannot retrieve an item absent from its candidate set.
4. **Bounded proof of concept:** a 12-query × 3-contrast-document setup (36 query–document pairs) can be used as a small initial test, **not a completed experiment**. Check tokenizer limits and truncation first.
5. **Targeted query expansion:** examine only remaining human-confirmed retrieval misses, separating lexical scorer effects from semantic expansion effects.

## Evaluation cautions

- `qDIR` changes the required answer; do not inherit `q0` relevance labels.
- Report candidate `K`, unique candidates, judged/unjudged coverage, and the exact evaluation denominator. With 60 documents, a genuine top-100 is not defined.
- Full-set recall cannot change when only reranking a frozen candidate set, but Recall@20 can change when reranking a 21-item set.
- The experimental BM25 phrase/alias scorer is not automatically equivalent to an Elasticsearch analyzer or synonym rewrite.
- No human adjudication is complete. Report conclusions as provisional synthetic-pilot observations.

## For a coding agent

Start by summarizing the available code tree, missing artifacts and inconsistencies. Propose **one bounded next task** and wait for operator approval before executing new experiments, downloading models, spending compute resources or changing baseline artifacts. Use distinct output IDs for new runs and report test results and limitations.

## Disclosure boundary

This draft intentionally omits organizational data, internal infrastructure, credentials, local machine paths, and previously excluded private reports. Review source ownership, employer disclosure rules and publication rights before distributing the handoff beyond authorized recipients.
