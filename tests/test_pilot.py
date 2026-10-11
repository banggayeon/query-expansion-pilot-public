"""Hand-computed tiny fixtures only; actual pilot corpus is NEVER searched."""
import copy
import math
import unittest
from dataclasses import replace

from qepilot.audit import audit, retrieval_documents
from qepilot.contracts import CandidateSnapshot, validate_same_candidates
from qepilot.evaluation import (Qrels, evaluate, evaluate_dir, evaluate_inv,
                                dir_sets, contrast_diagnostic, contrast_family)
from qepilot.reviews import import_reviews
from qepilot.scoring import ScoringConfig, bm25, phrase_tf, synonym_or
from qepilot.spans import Span, Alias, accept_aliases, query_groups, validate_spans
from qepilot.statistics import paired_summary, by_type, drift_summary
from qepilot.text import tokens, overlap


class SpanTests(unittest.TestCase):
    def setUp(self):
        self.query = "Keep TLS 1.2 only"
        self.spans = [Span(0, 4, "immutable_constraint", "keep"),
                      Span(5, 8, "alias_expandable_semantic", "transport_security"),
                      Span(9, 12, "immutable_constraint", "version_1_2"),
                      Span(13, 17, "immutable_constraint", "only")]
        self.alias = Alias(5, 8, "Transport Layer Security", "transport_security", True, "public-definition", "fixture-reviewer")

    def test_entity_alias_allowed_and_original_preserved(self):
        accepted, rejected = accept_aliases(self.query, self.spans, [self.alias])
        self.assertEqual(accepted, [self.alias])
        self.assertEqual(rejected, [])
        groups, _ = query_groups(self.query, self.spans, [self.alias])
        self.assertIn(("TLS", ("Transport Layer Security",)), groups)
        self.assertEqual(self.query, "Keep TLS 1.2 only")

    def test_immutable_version_cannot_expand(self):
        p = replace(self.alias, start=9, end=12, text="1.3", sense="version_1_2")
        a, r = accept_aliases(self.query, self.spans, [p])
        self.assertEqual(a, [])
        self.assertEqual(r[0]["reason"], "immutable_constraint")

    def test_changed_sense_rejected_even_when_attested(self):
        a, r = accept_aliases(self.query, self.spans, [replace(self.alias, sense="other_protocol")])
        self.assertEqual(a, [])
        self.assertEqual(r[0]["reason"], "sense_changed")

    def test_unverified_alias_rejected(self):
        for change in ({"verified": False}, {"source": ""}, {"verifier": ""}):
            with self.subTest(change=change):
                a, r = accept_aliases(self.query, self.spans, [replace(self.alias, **change)])
                self.assertFalse(a)
                self.assertEqual(r[0]["reason"], "unverified")

    def test_misaligned_source_rejected(self):
        _, r = accept_aliases(self.query, self.spans, [replace(self.alias, start=6)])
        self.assertEqual(r[0]["reason"], "source_span_missing")

    def test_alias_budget_and_dedup(self):
        proposals = [replace(self.alias, text=f"alternative {i}") for i in range(4)]
        a, r = accept_aliases(self.query, self.spans, proposals)
        self.assertEqual(len(a), 3)
        self.assertEqual(r[0]["reason"], "budget_exceeded")
        a, r = accept_aliases(self.query, self.spans, [self.alias, self.alias])
        self.assertEqual(len(a), 1)
        self.assertEqual(r[0]["reason"], "duplicate")

    def test_word_budget(self):
        _, r = accept_aliases(self.query, self.spans, [replace(self.alias, text="word " * 17)])
        self.assertEqual(r[0]["reason"], "budget_exceeded")

    def test_same_original_groups_for_O_and_D(self):
        o, _ = query_groups(self.query, self.spans)
        d, _ = query_groups(self.query, self.spans, [self.alias])
        self.assertEqual([x[0] for x in o], [x[0] for x in d])

    def test_overlapping_spans_error(self):
        with self.assertRaises(ValueError):
            validate_spans(self.query, self.spans + [Span(5, 12, "immutable_constraint", "bad")])

    def test_word_cut_error(self):
        with self.assertRaises(ValueError):
            validate_spans(self.query, [Span(6, 8, "alias_expandable_semantic", "bad")])


class ScoringTests(unittest.TestCase):
    def test_hand_bm25(self):
        # tf=1 and doc length=average => length-normalized factor is exactly 1.
        self.assertAlmostEqual(bm25(1, 1, 3, 10, 10), math.log(1 + 2.5 / 1.5))

    def test_zero_term_frequency(self):
        self.assertEqual(bm25(0, 0, 3, 10, 10), 0)

    def test_invalid_statistics(self):
        for args in ((1, 0, 3, 10, 10), (1, 4, 3, 10, 10), (1, 1, 3, 10, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                bm25(*args)

    def test_phrase_is_not_bag_of_words(self):
        self.assertEqual(phrase_tf(tokens("access control list"), "access control list"), 1)
        self.assertEqual(phrase_tf(tokens("access additional control list"), "access control list"), 0)
        self.assertEqual(phrase_tf(tokens("list control access"), "access control list"), 0)

    def test_phrase_repeated_occurrences(self):
        self.assertEqual(phrase_tf(tokens("a a a"), "a a"), 2)

    def test_weighted_max_not_sum(self):
        self.assertEqual(synonym_or([(2, [3, 3]), (1, [5])]), 9)
        self.assertEqual(synonym_or([(2, [3])]), synonym_or([(2, [3, 3])]))

    def test_config_discloses_experimental_implementation(self):
        self.assertEqual(ScoringConfig().manifest()["implementation"], "experimental_phrase_max_v1")

    def test_unsupported_tie_mode_rejected(self):
        with self.assertRaises(ValueError):
            synonym_or([(1, [2])], replace(ScoringConfig(), tie_contribution=.1))

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            synonym_or([(float("nan"), [])])


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.q = Qrels({"a": 2, "b": 1, "c": 0, "d": 2}, frozenset("abcd"), "complete_reviewed", "fixture-v1")

    def test_hand_recall_precision_ndcg(self):
        r = evaluate(list("acbd"), self.q, 2)
        self.assertEqual(r["recall"], .5)
        self.assertEqual(r["precision_known"], .5)
        self.assertAlmostEqual(r["ndcg"], 3 / (3 + 3 / math.log2(3)))
        self.assertEqual(r["judged_coverage"], 1)

    def test_relevance_cutoff(self):
        self.assertEqual(evaluate(list("bacd"), self.q, 1)["recall"], 0)
        self.assertAlmostEqual(evaluate(list("bacd"), self.q, 1, cutoff=1)["recall"], 1 / 3)

    def test_empty_relevance_is_not_zero_recall(self):
        q = Qrels({"a": 0}, frozenset("a"), "complete_reviewed", "v")
        r = evaluate(["a"], q, 1)
        self.assertIsNone(r["recall"])
        self.assertIsNone(r["ndcg"])

    def test_duplicate_unknown_underfill_rejected(self):
        for rank, k in ((list("aabc"), 2), (["x", "a"], 2), (["a"], 2)):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                evaluate(rank, self.q, k)

    def test_k_exceeds_corpus_is_na(self):
        r = evaluate(list("abcd"), self.q, 100)
        self.assertEqual(r["reason"], "K_exceeds_corpus")
        self.assertNotIn("recall", r)

    def test_draft_requires_explicit_opt_in(self):
        q = Qrels(self.q.grades, self.q.corpus_ids, "draft", "v")
        with self.assertRaises(ValueError):
            evaluate(list("abcd"), q, 2)
        self.assertEqual(evaluate(list("abcd"), q, 2, allow_draft=True)["interpretation"], "fixture_only")

    def test_cannot_claim_complete_with_partial_labels(self):
        with self.assertRaises(ValueError):
            Qrels({"a": 2}, frozenset("ab"), "complete_reviewed", "v")

    def test_partial_no_hit_unknown(self):
        q = Qrels({"a": 2}, frozenset("abcd"), "partial_reviewed", "v")
        r = evaluate(list("bcad"), q, 2)
        self.assertEqual(r["recall"], 0)  # explicitly diagnostic, not full recall
        self.assertEqual(r["interpretation"], "known_qrels_diagnostic")
        self.assertIsNone(r["success"])
        self.assertEqual(r["unjudged"], 2)

    def test_inv_paired_difference(self):
        a = evaluate(list("abcd"), self.q, 2)
        b = evaluate(list("adbc"), self.q, 2)
        self.assertEqual(evaluate_inv(a, b)["delta_recall"], .5)
        self.assertEqual(evaluate_inv(a, b)["both_success"], 1)

    def test_inv_snapshot_guard(self):
        a = evaluate(list("abcd"), self.q, 2)
        with self.assertRaises(ValueError):
            evaluate_inv(a, {**a, "snapshot_id": "other"})


class DirectionTests(unittest.TestCase):
    def setUp(self):
        ids = frozenset("abcd")
        self.old = Qrels({"a": 2, "b": 0, "c": 2, "d": 0}, ids, "complete_reviewed", "fixture")
        self.new = Qrels({"a": 0, "b": 2, "c": 2, "d": 0}, ids, "complete_reviewed", "fixture")

    def test_shared_excluded_and_rank_drop_positive(self):
        r = evaluate_dir(list("acbd"), list("bcda"), self.old, self.new, 2)
        self.assertEqual(r["old_only_hit"], 0)
        self.assertEqual(r["new_only_success"], 1)
        self.assertEqual(r["joint_success"], 1)
        self.assertEqual(r["old_only_mean_rank_drop"], 3)
        self.assertEqual(r["set_sizes"]["shared"], 1)

    def test_dropping_old_without_new_is_not_success(self):
        r = evaluate_dir(list("acbd"), list("dcab"), self.old, self.new, 1)
        self.assertEqual(r["joint_success"], 0)

    def test_one_sided_unjudged_does_not_create_oldonly(self):
        new = Qrels({"b": 2}, self.old.corpus_ids, "partial_reviewed", "fixture")
        s = dir_sets(self.old, new)
        self.assertEqual(s["old_only"], set())
        self.assertEqual(s["new_only"], {"b"})

    def test_partial_miss_is_unknown(self):
        new = Qrels({"a": 0, "b": 2}, self.old.corpus_ids, "partial_reviewed", "fixture")
        r = evaluate_dir(list("abcd"), list("cdab"), self.old, new, 2)
        self.assertIsNone(r["old_only_hit"])
        self.assertIsNone(r["new_only_success"])

    def test_partial_observed_hit_is_one(self):
        new = Qrels({"a": 0, "b": 2}, self.old.corpus_ids, "partial_reviewed", "fixture")
        r = evaluate_dir(list("abcd"), list("bcda"), self.old, new, 2)
        self.assertEqual(r["new_only_success"], 1)
        self.assertIsNone(r["old_only_hit"])

    def test_missing_rank_not_assigned_artificial_position(self):
        r = evaluate_dir(["c", "b"], ["b", "c"], self.old, self.new, 2)
        self.assertIsNone(r["old_only_mean_rank_drop"])

    def test_mismatched_snapshot_rejected(self):
        new = replace(self.new, snapshot_id="other")
        with self.assertRaises(ValueError):
            dir_sets(self.old, new)


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.q = Qrels({"a": 2, "b": 0, "h": 0}, frozenset("abhx"), "partial_reviewed", "fixture")

    def test_complete_contrast_partial_global_qrels(self):
        r = contrast_diagnostic({"a": 3, "b": 2, "h": 1}, self.q, list("abh"), "h")
        self.assertEqual(r["pair_accuracy"], 1)
        self.assertEqual(r["top1_positive"], 1)
        self.assertEqual(r["global_qrels_status"], "partial_reviewed")
        self.assertEqual(r["scope"], "contrast_only")

    def test_hard_negative_failure(self):
        r = contrast_diagnostic({"a": 2, "b": 1, "h": 3}, self.q, list("abh"), "h")
        self.assertEqual(r["pair_accuracy"], .5)
        self.assertEqual(r["hard_negative_before_positive"], 1)
        self.assertEqual(r["top1_positive"], 0)

    def test_tie_counts_half(self):
        r = contrast_diagnostic({"a": 1, "b": 1, "h": 0}, self.q, list("abh"), "h")
        self.assertEqual(r["pair_accuracy"], .75)
        self.assertEqual(r["top1_positive"], .5)

    def test_missing_label_pending(self):
        q = replace(self.q, grades={"a": 2, "b": 0})
        self.assertEqual(contrast_diagnostic({}, q, list("abh"), "h")["status"], "pending_labels")

    def test_missing_score_unknown(self):
        self.assertEqual(contrast_diagnostic({"a": 2}, self.q, list("abh"), "h")["status"], "unknown_scores")

    def test_draft_contrast_not_promoted(self):
        q = replace(self.q, status="draft")
        with self.assertRaises(ValueError):
            contrast_diagnostic({"a": 3, "b": 2, "h": 1}, q, list("abh"), "h")

    def test_all_three_variants_independently_required(self):
        qnew = replace(self.q, grades={"a": 0, "b": 2, "h": 0})
        scores = {"o": {"a": 3, "b": 2, "h": 1}, "i": {"a": 3, "b": 1, "h": 0}, "d": {"a": 1, "b": 3, "h": 0}}
        roles = {"original_relevant": "a", "changed_relevant": "b", "hard_negative": "h"}
        r = contrast_family(scores, {"o": self.q, "i": self.q, "d": qnew}, {"q0": "o", "qINV": "i", "qDIR": "d"}, roles)
        self.assertEqual(r["strict_positive_switch"], 1)
        pending_inv = replace(self.q, grades={})
        r = contrast_family(scores, {"o": self.q, "i": pending_inv, "d": qnew}, {"q0": "o", "qINV": "i", "qDIR": "d"}, roles)
        self.assertIsNone(r["strict_positive_switch"])


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.docs = {"a": {"title": "A", "text": "The counter uses fifteen minutes."}, "b": {"title": "B", "text": "Other topic."}}
        self.row = {"query_id": "q", "doc_id": "a", "status": "adjudicated", "grade": 2,
                    "reviews": [{"reviewer_id": who, "reviewer_kind": "human", "independent": True,
                                 "grade": 2, "evidence": "fifteen minutes", "reason": "matches required window"}
                                for who in ("fixture-human-1", "fixture-human-2")],
                    "adjudication": {"reviewer_id": "fixture-human-1", "grade": 2, "reason": "agreement"}}

    def test_independent_evidence_import(self):
        q = import_reviews("q", [self.row], self.docs, "fixture")
        self.assertEqual(q.grades, {"a": 2})
        self.assertEqual(q.status, "partial_reviewed")

    def test_author_cannot_self_certify(self):
        row = copy.deepcopy(self.row)
        row["reviews"][0]["reviewer_id"] = "assistant_author"
        with self.assertRaises(ValueError):
            import_reviews("q", [row], self.docs, "fixture")

    def test_no_fake_human_via_kind(self):
        row = copy.deepcopy(self.row)
        row["reviews"][0]["reviewer_kind"] = "model"
        with self.assertRaises(ValueError):
            import_reviews("q", [row], self.docs, "fixture")

    def test_duplicate_reviewer_rejected(self):
        row = copy.deepcopy(self.row)
        row["reviews"][1] = row["reviews"][0]
        with self.assertRaises(ValueError):
            import_reviews("q", [row], self.docs, "fixture")

    def test_unseen_evidence_rejected(self):
        row = copy.deepcopy(self.row)
        row["reviews"][0]["evidence"] = "sixty minutes"
        with self.assertRaises(ValueError):
            import_reviews("q", [row], self.docs, "fixture")

    def test_unjudged_remains_absent(self):
        q = import_reviews("q", [{"query_id": "q", "doc_id": "a", "grade": None, "status": "unjudged"}], self.docs, "fixture")
        self.assertEqual(q.grades, {})


class StatisticsTests(unittest.TestCase):
    def rows(self, n=5):
        meta = {"snapshot_id": "fixture", "k": 10, "qrels_status": "partial_reviewed", "interpretation": "known_qrels_diagnostic"}
        return [{"family_id": f"f{i}", "variant": v, "before": {**meta, "recall": .2}, "after": {**meta, "recall": .4},
                 "snapshot_id": "fixture", "comparison": "D/O-O/O", "type": "lexical"}
                for i in range(n) for v in ("q0", "qINV", "qDIR")]

    def test_family_not_query_is_sample(self):
        r = paired_summary(self.rows())
        self.assertEqual(r["n_families"], 5)
        self.assertAlmostEqual(r["mean_delta"], .2)
        self.assertEqual(r["ci95"], [.2, .2])

    def test_unknown_excludes_whole_family(self):
        rows = self.rows()
        rows[0]["after"]["recall"] = None
        r = paired_summary(rows)
        self.assertEqual(r["n_families"], 4)
        self.assertEqual(r["excluded_unknown"], ["f0"])
        self.assertIsNone(r["ci95"])

    def test_unpaired_family_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary(self.rows()[:-1])

    def test_mixed_snapshots_rejected(self):
        rows = self.rows()
        rows[0]["snapshot_id"] = "another"
        with self.assertRaises(ValueError):
            paired_summary(rows)

    def test_mixed_qrels_status_rejected(self):
        rows = self.rows()
        rows[0]["after"]["qrels_status"] = "complete_reviewed"
        with self.assertRaises(ValueError):
            paired_summary(rows)

    def test_mixed_k_across_families_rejected(self):
        rows = self.rows()
        rows[0]["before"]["k"] = 20
        rows[0]["after"]["k"] = 20
        with self.assertRaises(ValueError):
            paired_summary(rows)

    def test_type_report_preserves_overall(self):
        r = by_type(self.rows(2), "type")
        self.assertEqual(r["overall"]["n_families"], 2)
        self.assertEqual(r["by_type"]["lexical"]["n_families"], 2)

    def test_drift_and_loss_are_independent(self):
        r = drift_summary([{"drift": True, "expanded": True, "delta": .1},
                           {"drift": False, "expanded": True, "delta": -.1},
                           {"drift": None, "expanded": True, "delta": -.2}])
        self.assertEqual(r["drift_rate"], .5)
        self.assertEqual(r["drift_loss_table"]["drift_True_loss_False"], 1)
        self.assertEqual(r["drift_loss_table"]["drift_False_loss_True"], 1)
        self.assertEqual(r["unknown_n"], 1)


class DataAndContractTests(unittest.TestCase):
    def test_dataset_structure_only_no_retrieval(self):
        report = audit()
        self.assertEqual(report["documents"], 60)
        self.assertEqual(report["queries"], 36)
        self.assertEqual(report["contrast_required_cells"], 36)
        self.assertEqual(report["human_reviewed_cells"], 0)
        self.assertFalse(report["cpu_search_executed"])

    def test_metadata_not_exposed_to_retrieval(self):
        docs = retrieval_documents([{"doc_id": "d", "title": "T", "text": "X", "source_family": "f", "grade": 2}])
        self.assertEqual(set(docs[0]), {"doc_id", "title", "text"})

    def test_tokenizer_protects_versions_negation_units(self):
        self.assertEqual(tokens("NOT TLS 1.2 >= 15-minute IPv6-only"), ("not", "tls", "1.2", ">=", "15-minute", "ipv6-only"))

    def test_overlap_is_not_semantic_label(self):
        r = overlap("RPO", "RTO")
        self.assertEqual(r["token_jaccard"], 0)
        self.assertAlmostEqual(r["character_similarity"], 2 / 3)

    def test_reranker_candidate_identity_contract(self):
        s = CandidateSnapshot("c", "q", ("a", "b"), "set", "order", "D/O", 2, "corpus", "qrels")
        validate_same_candidates(s, ["b", "a"])
        for ids in (["a"], ["a", "a"], ["a", "c"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_same_candidates(s, ids)


if __name__ == "__main__":
    unittest.main()
