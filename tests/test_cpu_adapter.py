"""Tiny adapter checks; frozen evaluation/scoring and prior tests are unchanged."""
import unittest
from experiments.run_minimal_cpu import make_expansions, term_statistics, lexical_scores, rank_scores, rrf, validate_qc, ROOT
from qepilot.audit import read_jsonl


class CPUAdapterTests(unittest.TestCase):
    def test_dictionary_and_original_use_same_groups(self):
        query = {"query_id": "q", "family_id": "f", "variant": "q0", "text": "TLS",
                 "spans": [{"start": 0, "end": 3, "kind": "alias_expandable_semantic", "sense": "tls"}]}
        dictionary = [{"term": "TLS", "alias": "Transport Layer Security", "sense": "tls", "status": "source_checked", "source_id": "s"}]
        o, d, logs = make_expansions([query], dictionary, {"s": {"url": "fixture_source"}})
        self.assertEqual(o["q"][0][0], d["q"][0][0])
        self.assertTrue(logs[0]["expanded"])
        self.assertFalse(logs[0]["human_reviewed"])
        docs = [{"doc_id": "a", "title": "", "text": "TLS"},
                {"doc_id": "b", "title": "", "text": "Transport Layer Security"}]
        index = term_statistics(docs, [o, d])
        self.assertEqual(lexical_scores(o, index)["q"]["b"], 0)
        self.assertGreater(lexical_scores(d, index)["q"]["b"], 0)

    def test_no_same_sense_means_no_expansion(self):
        q = {"query_id": "q", "family_id": "f", "variant": "q0", "text": "ACL",
             "spans": [{"start": 0, "end": 3, "kind": "alias_expandable_semantic", "sense": "other"}]}
        d = [{"term": "ACL", "alias": "access control list", "sense": "network", "status": "source_checked", "source_id": "s"}]
        o, expanded, logs = make_expansions([q], d, {"s": {"url": "fixture"}})
        self.assertEqual(o, expanded)
        self.assertFalse(logs[0]["expanded"])

    def test_rrf_uses_full_unique_corpus(self):
        s = rrf(["a", "b", "c"], ["b", "a", "c"])
        self.assertAlmostEqual(s["a"], 1 / 61 + 1 / 62)
        self.assertEqual(rank_scores(s), ["a", "b", "c"])

    def test_qc_evidence_and_coverage(self):
        rows, summary = validate_qc(ROOT, read_jsonl(ROOT / "data/queries.jsonl"),
                                   read_jsonl(ROOT / "data/corpus.jsonl"), read_jsonl(ROOT / "data/qrels.draft.jsonl"))
        self.assertEqual(len(rows), 36)
        self.assertEqual(summary["human_reviews"], 0)
        self.assertEqual(summary["ambiguous_cells"], 3)


if __name__ == "__main__":
    unittest.main()
