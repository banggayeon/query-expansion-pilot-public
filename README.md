# Query Expansion Pilot (Public Release Candidate)

An independent, small-scale research pilot exploring query expansion, lexical ambiguity, BM25 scoring, and evaluation of synthetic retrieval tasks.

**Publication status:** Prepared for review. This repository is private until the owner confirms distribution rights and approves publication.

## Scope
- Synthetic query/document data and draft relevance judgments
- Standalone scoring/evaluation modules and tests
- Lightweight experimental scripts and portable configuration

## Limitations
All relevance judgments are draft/provisional. The pilot does not establish production performance or any organization's search quality.

## Reproducibility
Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m qepilot.audit
```

Additional experiments may need third-party Python packages or publicly available model weights. Some original machine-specific execution artifacts and results are intentionally not distributed. Avoid treating missing run output as precomputed evidence.

## Source and disclosure review
This distribution intentionally excludes local machine paths, infrastructure inventory, and internal-integration planning materials. No original Git history is transferred. Before changing visibility to public, independently confirm that the code, authored synthetic datasets, and associated work product may be distributed under all applicable agreements.

No open-source license is granted by this repository unless a license is added by the owner.
