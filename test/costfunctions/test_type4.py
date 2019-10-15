import math
from unittest import TestCase

from src.costfunctions.type1 import Type1
from src.costfunctions.type4 import Type4
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import separated_cosine_similarity, combined_cosine_similarity
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_generator import DataGenerator


class TestType4(TestCase):
    def test_instantiation(self):
        t4 = Type4(euclidean_distance, separated_cosine_similarity, (1 / 3), (1 / 3), (1 / 3), math.inf, 1, 0.5, 0.6,
                   0.7, False)
        self.assertEqual(euclidean_distance.__get__, t4.distance_metric.__get__)
        self.assertEqual(separated_cosine_similarity.__get__, t4.similarity_metric.__get__)
        self.assertAlmostEqual(t4.alpha, (1 / 3), delta=0.01)
        self.assertAlmostEqual(t4.beta, (1 / 3), delta=0.01)
        self.assertAlmostEqual(t4.omega, (1 / 3), delta=0.01)
        self.assertAlmostEqual(t4.phi_1, math.inf, delta=0.01)
        self.assertAlmostEqual(t4.phi_2, 1.0, delta=0.01)
        self.assertAlmostEqual(t4.query_distance_threshold, 0.5, delta=0.01)
        self.assertAlmostEqual(t4.dataset_distance_threshold, 0.6, delta=0.01)
        self.assertAlmostEqual(t4.keyword_similarity_threshold, 0.7, delta=0.01)
        self.assertEqual(t4.disable_thresholds, False)

    def test_same_as_type1(self):
        # TODO These parameters should according to the paper transform the type4 cost function so that it equals the type1 cost function.
        # TODO However, it does not reliably solve for the same results. This probably means there is something wrong with the type4 cost function.
        alpha = 1 / 3
        beta = 1 / 3
        omega = 1 / 3
        phi1 = math.inf
        phi2 = 1
        t1 = Type1(euclidean_distance, combined_cosine_similarity, alpha, beta, omega)
        t4 = Type4(euclidean_distance, combined_cosine_similarity, alpha, beta, omega, phi1, phi2)
        possible_keywords = '1 2 3 4 5 6 7 8 9 0'.split()
        dg = DataGenerator(possible_keywords)
        query = dg.generate(1)[0]
        data = dg.generate(5)
        ns1 = NaiveSolver(query, data, t1)
        ns4 = NaiveSolver(query, data, t4)
        result1 = ns1.solve()
        result4 = ns4.solve()
        self.assertListEqual(result1, result4)
