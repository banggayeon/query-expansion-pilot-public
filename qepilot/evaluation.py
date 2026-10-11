"""Metrics over supplied rankings only. Never generates a ranking or calls a model."""
from dataclasses import dataclass
import math


@dataclass
class Qrels:
    grades: dict
    corpus_ids: frozenset
    status: str
    snapshot_id: str

    def __post_init__(self):
        if self.status not in {"draft", "partial_reviewed", "complete_reviewed"}:
            raise ValueError("invalid qrels status")
        if not self.snapshot_id or not self.corpus_ids or not set(self.grades) <= self.corpus_ids:
            raise ValueError("invalid qrels IDs")
        if any(type(g) is not int or g not in {0, 1, 2} for g in self.grades.values()):
            raise ValueError("grades must be 0/1/2; unjudged is absent")
        if self.status == "complete_reviewed" and set(self.grades) != self.corpus_ids:
            raise ValueError("complete qrels require entire corpus")

    def guard(self, allow_draft=False):
        if self.status == "draft" and not allow_draft:
            raise ValueError("draft labels cannot be scored without explicit fixture opt-in")


def validate_ranking(ranking, corpus_ids):
    if len(ranking) != len(set(ranking)) or not set(ranking) <= corpus_ids:
        raise ValueError("duplicate or unknown ranked document")


def _top(ranking, corpus_ids, k):
    if type(k) is not int or k <= 0:
        raise ValueError("K must be a positive integer")
    validate_ranking(ranking, corpus_ids)
    if k > len(corpus_ids):
        return None
    if len(ranking) < k:
        raise ValueError("candidate budget underfilled")
    return ranking[:k]


def evaluate(ranking, qrels, k, *, cutoff=2, allow_draft=False):
    qrels.guard(allow_draft)
    if cutoff not in {1, 2}:
        raise ValueError("invalid cutoff")
    top = _top(ranking, qrels.corpus_ids, k)
    scope = {"draft": "fixture_only", "partial_reviewed": "known_qrels_diagnostic",
             "complete_reviewed": "complete_corpus"}[qrels.status]
    result = {"k": k, "qrels_status": qrels.status, "snapshot_id": qrels.snapshot_id,
              "interpretation": scope, "cutoff": cutoff}
    if top is None:
        return {**result, "status": "not_applicable", "reason": "K_exceeds_corpus"}
    relevant = {d for d, g in qrels.grades.items() if g >= cutoff}
    hits = len(set(top) & relevant)
    judged = sum(d in qrels.grades for d in top)
    dcg = sum((2 ** qrels.grades.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sorted(qrels.grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    success = 1 if hits else (0 if relevant and judged == k else None)
    return {**result, "status": "ok", "recall": hits / len(relevant) if relevant else None,
            "ndcg": dcg / idcg if idcg else None, "precision_known": hits / k,
            "success": success, "success_known": int(hits > 0) if relevant else None,
            "mrr": next((1 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0) if relevant else None,
            "judged_coverage": judged / k, "judged": judged, "unjudged": k - judged,
            "known_relevant": len(relevant)}


def dir_sets(old, new, cutoff=2):
    if old.corpus_ids != new.corpus_ids or old.snapshot_id != new.snapshot_id:
        raise ValueError("DIR requires same corpus and qrels snapshot")
    if cutoff not in {1, 2}:
        raise ValueError("invalid cutoff")
    common = set(old.grades) & set(new.grades)
    a = {d for d in common if old.grades[d] >= cutoff}
    b = {d for d in common if new.grades[d] >= cutoff}
    return {"old_only": a - b, "new_only": b - a, "shared": a & b, "jointly_judged": common}


def evaluate_dir(old_ranking, changed_ranking, old, new, k=10, *, cutoff=2, allow_draft=False):
    old.guard(allow_draft)
    new.guard(allow_draft)
    sets = dir_sets(old, new, cutoff)
    before, after = _top(old_ranking, old.corpus_ids, k), _top(changed_ranking, new.corpus_ids, k)
    if before is None or after is None:
        return {"status": "not_applicable", "reason": "K_exceeds_corpus", "k": k}
    jointly_covered = set(after) <= sets["jointly_judged"]

    def hit(target):
        if not target:
            return None
        if set(after) & target:
            return 1
        return 0 if jointly_covered else None

    old_hit, new_success = hit(sets["old_only"]), hit(sets["new_only"])
    ranks_a = {d: i + 1 for i, d in enumerate(old_ranking)}
    ranks_b = {d: i + 1 for i, d in enumerate(changed_ranking)}
    targets = sets["old_only"]
    drop = (sum(ranks_b[d] - ranks_a[d] for d in targets) / len(targets)
            if targets and targets <= ranks_a.keys() and targets <= ranks_b.keys() else None)
    return {"status": "ok", "k": k,
            "qrels_status": [old.status, new.status], "snapshot_id": old.snapshot_id,
            "interpretation": ("fixture_only" if "draft" in (old.status, new.status) else
                               "complete_corpus" if old.status == new.status == "complete_reviewed" else
                               "known_qrels_diagnostic"),
            "old_only_hit": old_hit, "new_only_success": new_success,
            "old_only_intrusion_known": len(set(after) & targets) / k,
            "joint_success": int(new_success == 1 and old_hit == 0) if old_hit is not None and new_success is not None else None,
            "old_only_mean_rank_drop": drop,
            "joint_coverage": len(set(after) & sets["jointly_judged"]) / k,
            "set_sizes": {key: len(value) for key, value in sets.items()}}


def evaluate_inv(original_metrics, inv_metrics):
    for key in ("snapshot_id", "k", "qrels_status"):
        if original_metrics.get(key) != inv_metrics.get(key):
            raise ValueError("INV metrics must share snapshot, K and status")
    result = {}
    for name in ("recall", "ndcg", "success"):
        a, b = original_metrics.get(name), inv_metrics.get(name)
        result[f"delta_{name}"] = b - a if a is not None and b is not None else None
    a, b = original_metrics.get("success"), inv_metrics.get("success")
    result["both_success"] = int(a == b == 1) if a is not None and b is not None else None
    return result


def contrast_diagnostic(scores, qrels, contrast_ids, hard_negative_id, *, allow_draft=False):
    """Closed-set discrimination, independent of GLOBAL qrels completeness.

    All contrast labels must be reviewed (or explicitly opted-in draft fixture),
    all contrast scores observed. Larger score is better; ties count half.
    Roles never supply labels: positives are taken only from per-query grades.
    """
    qrels.guard(allow_draft)
    ids = set(contrast_ids)
    if len(ids) != 3 or hard_negative_id not in ids or not ids <= qrels.corpus_ids:
        raise ValueError("contrast requires three distinct known documents")
    base = {"scope": "contrast_only", "global_qrels_status": qrels.status,
            "reviewed": qrels.status != "draft", "label_coverage": len(ids & qrels.grades.keys()) / 3}
    if not ids <= qrels.grades.keys():
        return {**base, "status": "pending_labels"}
    if not ids <= scores.keys():
        return {**base, "status": "unknown_scores"}
    if any(not math.isfinite(scores[d]) for d in ids):
        raise ValueError("non-finite score")
    positive = {d for d in ids if qrels.grades[d] == 2}
    negative = {d for d in ids if qrels.grades[d] == 0}
    if not positive or not negative or qrels.grades[hard_negative_id] != 0:
        return {**base, "status": "invalid_contrast_labels"}

    def better(a, b):
        return 1.0 if scores[a] > scores[b] else (0.5 if scores[a] == scores[b] else 0.0)

    pairs = [better(p, n) for p in positive for n in negative]
    best = max(scores[d] for d in ids)
    tied = {d for d in ids if scores[d] == best}
    return {**base, "status": "ok", "pair_accuracy": sum(pairs) / len(pairs),
            "top1_positive": len(tied & positive) / len(tied),
            "hard_negative_before_positive": sum(better(hard_negative_id, p) for p in positive) / len(positive),
            "positive_ids": sorted(positive), "pair_count": len(pairs)}


def contrast_family(scores_by_query, qrels_by_query, query_ids, roles, *, allow_draft=False):
    if set(query_ids) != {"q0", "qINV", "qDIR"}:
        raise ValueError("all three variants required")
    if set(roles) != {"original_relevant", "changed_relevant", "hard_negative"}:
        raise ValueError("missing contrast role")
    if len(set(query_ids.values())) != 3:
        raise ValueError("independent query IDs required")
    qs = [qrels_by_query[q] for q in query_ids.values()]
    if any(q.corpus_ids != qs[0].corpus_ids or q.snapshot_id != qs[0].snapshot_id for q in qs):
        raise ValueError("contrast requires same corpus and snapshot")
    out = {v: contrast_diagnostic(scores_by_query.get(q, {}), qrels_by_query[q],
                                  roles.values(), roles["hard_negative"], allow_draft=allow_draft)
           for v, q in query_ids.items()}
    switch = None
    if all(x["status"] == "ok" for x in out.values()):
        a, b = roles["original_relevant"], roles["changed_relevant"]
        expected = {"q0": a, "qINV": a, "qDIR": b}
        valid_roles = all(expected[v] in out[v]["positive_ids"] and
                          qrels_by_query[query_ids[v]].grades[b if v != "qDIR" else a] == 0
                          for v in expected)
        if valid_roles:
            switch = int(all(scores_by_query[query_ids[v]][expected[v]] >
                             scores_by_query[query_ids[v]][b if v != "qDIR" else a] for v in expected))
    return {"variants": out, "strict_positive_switch": switch, "scope": "contrast_only"}
