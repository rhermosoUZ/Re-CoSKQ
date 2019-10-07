from unittest import TestCase

from src.costfunctions.type1 import Type1
from src.costfunctions.type2 import Type2
from src.costfunctions.type3 import Type3
from src.evaluator import Evaluator
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import combined_cosine_similarity
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_generator import DataGenerator


class TestEvaluator(TestCase):
    def test_general(self):
        ev = Evaluator()
        possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
        dg = DataGenerator(possible_keywords)
        gen_query = dg.generate(1)[0]
        gen_data = dg.generate(10)
        cf1 = Type1(euclidean_distance, combined_cosine_similarity, 0.33, 0.33, 0.33, disable_thresholds=True)
        cf2 = Type2(euclidean_distance, combined_cosine_similarity, 0.33, 0.33, 0.33, disable_thresholds=True)
        cf3 = Type3(euclidean_distance, combined_cosine_similarity, 0.33, 0.33, 0.33, disable_thresholds=True)
        ns1 = NaiveSolver(gen_query, gen_data, cf1, result_length=10, max_subset_size=6)
        ns2 = NaiveSolver(gen_query, gen_data, cf2, result_length=10, max_subset_size=6)
        ns3 = NaiveSolver(gen_query, gen_data, cf3, result_length=10, max_subset_size=6)
        ev.add_solver(ns1)
        ev.add_solver(ns2)
        ev.add_solver(ns3)
        ev.evaluate()
        results = ev.get_results()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[1]), 2)
        self.assertEqual(len(results[2]), 2)
        self.assertEqual(len(results[0][0]), 10)
        self.assertEqual(len(results[1][0]), 10)
        self.assertEqual(len(results[2][0]), 10)
