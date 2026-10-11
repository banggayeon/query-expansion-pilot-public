"""Explicit human-review import. Generated proposed grades are never imported."""
from .evaluation import Qrels


def import_reviews(query_id, records, documents, snapshot_id, *, complete=False, min_reviewers=2):
    if min_reviewers < 1:
        raise ValueError("at least one independent reviewer required")
    grades, seen = {}, set()
    for row in records:
        if row["query_id"] != query_id:
            continue
        doc_id = row["doc_id"]
        if doc_id not in documents or doc_id in seen:
            raise ValueError("unknown or duplicate review cell")
        seen.add(doc_id)
        if row["status"] == "unjudged":
            if row.get("grade") is not None:
                raise ValueError("unjudged cell contains grade")
            continue
        if row["status"] != "adjudicated":
            raise ValueError("review must be adjudicated before import")
        reviews = row.get("reviews", [])
        people = {r.get("reviewer_id") for r in reviews}
        if (len(people) < min_reviewers or len(people) != len(reviews)
                or None in people or "" in people or "assistant_author" in people):
            raise ValueError("independent distinct reviewers required")
        body = documents[doc_id]["title"] + " " + documents[doc_id]["text"]
        for r in reviews:
            if (r.get("reviewer_kind") != "human" or r.get("independent") is not True
                    or type(r.get("grade")) is not int or r["grade"] not in {0, 1, 2}
                    or not r.get("evidence") or r["evidence"] not in body or not r.get("reason")):
                raise ValueError("missing review evidence, grade, or independence attestation")
        final = row.get("adjudication") or {}
        grade = final.get("grade")
        if (type(grade) is not int or grade not in {0, 1, 2} or row.get("grade") != grade
                or not final.get("reviewer_id") or not final.get("reason")):
            raise ValueError("missing final adjudication")
        grades[doc_id] = grade
    return Qrels(grades, frozenset(documents), "complete_reviewed" if complete else "partial_reviewed", snapshot_id)
