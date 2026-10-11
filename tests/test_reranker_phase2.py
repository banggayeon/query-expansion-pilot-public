import importlib.util
from pathlib import Path
import unittest
from qepilot.contracts import CandidateSnapshot,validate_same_candidates
from qepilot.evaluation import Qrels,evaluate

s=importlib.util.spec_from_file_location('worker',Path(__file__).resolve().parents[1]/'experiments/qwen_contrast_worker.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class RerankerPhase2Tests(unittest.TestCase):
    def test_stable_probability(self):
        self.assertEqual(w.probability(0),.5);self.assertEqual(w.probability(1000),1);self.assertEqual(w.probability(-1000),0)
        with self.assertRaises(ValueError):w.probability(float('nan'))
    def test_contract(self):
        s=CandidateSnapshot('c','q',('a','b','c'),'h','o','fixture',3,'c','r')
        validate_same_candidates(s,['c','b','a'])
        for bad in [['a','b'],['a','b','b'],['a','b','d']]:
            with self.assertRaises(ValueError):validate_same_candidates(s,bad)
    def test_recall20_invariance(self):
        docs=[str(i) for i in range(20)];q=Qrels({'0':2},frozenset(docs),'draft','r')
        self.assertEqual(evaluate(docs,q,20,allow_draft=True)['recall'],evaluate(docs[::-1],q,20,allow_draft=True)['recall'])
        self.assertNotEqual(evaluate(docs,q,10,allow_draft=True)['mrr'],evaluate(docs[::-1],q,10,allow_draft=True)['mrr'])
    def test_union21_exception(self):
        docs=[str(i) for i in range(21)];q=Qrels({'0':2},frozenset(docs),'draft','r')
        self.assertNotEqual(evaluate(docs,q,20,allow_draft=True)['recall'],evaluate(docs[::-1],q,20,allow_draft=True)['recall'])
        self.assertEqual(evaluate(docs,q,21,allow_draft=True)['recall'],evaluate(docs[::-1],q,21,allow_draft=True)['recall'])
