"""Data contracts for a future reranker adapter; no reranker implementation."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_set_id: str
    query_id: str
    ordered_doc_ids: tuple[str, ...]
    set_hash: str
    order_hash: str
    candidate_origin: str
    budget: int
    corpus_hash: str
    qrels_snapshot_id: str


class FutureReranker(Protocol):
    def rerank(self, query_input: str, snapshot: CandidateSnapshot, config: dict) -> list[tuple[str, float]]:
        """Must return exactly the same unique document set; not implemented."""
        ...


def validate_same_candidates(snapshot, after_ids):
    if len(snapshot.ordered_doc_ids) != snapshot.budget or len(set(snapshot.ordered_doc_ids)) != snapshot.budget:
        raise ValueError("invalid snapshot budget")
    if len(after_ids) != snapshot.budget or set(after_ids) != set(snapshot.ordered_doc_ids):
        raise ValueError("reranker added, dropped or duplicated candidates")
