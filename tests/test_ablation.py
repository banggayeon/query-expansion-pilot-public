import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('ablation',Path(__file__).resolve().parents[1]/'experiments/bm25_ablation.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AblationTests(unittest.TestCase):
    def test_numeric_forms(self):
        for t in ('15-minute','fifteen-minute','fifteen minutes'):
            self.assertEqual(a.normalize(a.tokens(t)),('15','minute'))
        self.assertEqual(a.normalize(a.tokens('twenty-one hours twenty one hours')),('21','hour','21','hour'))
    def test_constraints(self):
        seq=a.tokens('not no without only 1.2 >= 100 milliseconds')
        self.assertEqual(a.normalize(seq),seq)
        for t in seq:self.assertFalse(a.removable(t,set()))
    def test_protected_and_phrase(self):
        self.assertTrue(a.removable('how',set()))
        self.assertFalse(a.removable('how',{('how',)}))
        self.assertFalse(a.removable('identity of a service account',set()))
        self.assertIn(('of',),a.units('identity of a service account',0,1))
    def test_repeated_sum_units(self):
        self.assertEqual(a.units('a a',0,1),[('a',),('a',)])
        self.assertEqual(a.units('a a',0,0),[('a','a')])
    def test_transitions(self):
        self.assertEqual(a.transition('below','above'),'correction')
        self.assertEqual(a.transition('above','below'),'regression')
        self.assertEqual(a.transition('above','tie'),'to_tie')
        self.assertEqual(a.transition('tie','above'),'from_tie_above')
        self.assertEqual(a.transition('below','below'),'unchanged_below')
