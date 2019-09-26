from unittest import TestCase

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import separated_cosine_similarity, combined_cosine_similarity
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.solver import Solver


class TestSolver(TestCase):
    def test_instantiation(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False, result_length=10)
        self.assertAlmostEqual(so.query.coordinates.x, 0, delta=0.01)
        self.assertAlmostEqual(so.query.coordinates.y, 0, delta=0.01)
        self.assertListEqual(so.data, data)
        self.assertAlmostEqual(so.data[0].coordinates.x, 1, delta=0.01)
        self.assertAlmostEqual(so.data[0].coordinates.y, 1, delta=0.01)
        self.assertListEqual(so.data[0].keywords, kwc1_keywords)
        for index in range(len(so.data[0].keywords)):
            self.assertEqual(so.data[0].keywords[index], kwc1_keywords[index])
        self.assertAlmostEqual(so.data[1].coordinates.x, 2, delta=0.01)
        self.assertAlmostEqual(so.data[1].coordinates.y, 2, delta=0.01)
        self.assertListEqual(so.data[1].keywords, kwc2_keywords)
        for index in range(len(so.data[1].keywords)):
            self.assertEqual(so.data[1].keywords[index], kwc2_keywords[index])
        self.assertAlmostEqual(so.data[2].coordinates.x, 3, delta=0.01)
        self.assertAlmostEqual(so.data[2].coordinates.y, 3, delta=0.01)
        self.assertListEqual(so.data[2].keywords, kwc3_keywords)
        for index in range(len(so.data[2].keywords)):
            self.assertEqual(so.data[2].keywords[index], kwc3_keywords[index])
        self.assertEqual(euclidean_distance.__get__, so.cost_function.distance_metric.__get__)
        self.assertEqual(separated_cosine_similarity.__get__, so.cost_function.similarity_metric.__get__)
        self.assertAlmostEqual(so.cost_function.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(so.cost_function.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(so.cost_function.omega, 0.4, delta=0.01)
        self.assertEqual(so.normalize_data, False)
        self.assertEqual(so.result_length, 10)
        self.assertAlmostEqual(so.denormalize_max_x, 0.0, delta=0.01)
        self.assertAlmostEqual(so.denormalize_min_x, 0.0, delta=0.01)
        self.assertAlmostEqual(so.denormalize_max_y, 0.0, delta=0.01)
        self.assertAlmostEqual(so.denormalize_min_y, 0.0, delta=0.01)

    def test_get_max_inter_dataset_distance(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False)
        fs1 = frozenset([kwc1])
        fs2 = frozenset([kwc2])
        fs3 = frozenset([kwc3])
        fs4 = frozenset([kwc1, kwc2])
        fs5 = frozenset([kwc1, kwc3])
        fs6 = frozenset([kwc2, kwc3])
        fs7 = frozenset([kwc1, kwc2, kwc3])
        result = so.get_max_inter_dataset_distance()
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result.get(fs1), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs2), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs3), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs4), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs5), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs6), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs7), 2.83, delta=0.01)

    def test_get_min_inter_dataset_distance(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False)
        fs1 = frozenset([kwc1])
        fs2 = frozenset([kwc2])
        fs3 = frozenset([kwc3])
        fs4 = frozenset([kwc1, kwc2])
        fs5 = frozenset([kwc1, kwc3])
        fs6 = frozenset([kwc2, kwc3])
        fs7 = frozenset([kwc1, kwc2, kwc3])
        result = so.get_min_inter_dataset_distance()
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result.get(fs1), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs2), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs3), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs4), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs5), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs6), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs7), 1.41, delta=0.01)

    def test_get_max_query_dataset_distance(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False)
        fs1 = frozenset([kwc1])
        fs2 = frozenset([kwc2])
        fs3 = frozenset([kwc3])
        fs4 = frozenset([kwc1, kwc2])
        fs5 = frozenset([kwc1, kwc3])
        fs6 = frozenset([kwc2, kwc3])
        fs7 = frozenset([kwc1, kwc2, kwc3])
        result = so.get_max_query_dataset_distance()
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result.get(fs1), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs2), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs3), 4.24, delta=0.01)
        self.assertAlmostEqual(result.get(fs4), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs5), 4.24, delta=0.01)
        self.assertAlmostEqual(result.get(fs6), 4.24, delta=0.01)
        self.assertAlmostEqual(result.get(fs7), 4.24, delta=0.01)

    def test_get_min_query_dataset_distance(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False)
        fs1 = frozenset([kwc1])
        fs2 = frozenset([kwc2])
        fs3 = frozenset([kwc3])
        fs4 = frozenset([kwc1, kwc2])
        fs5 = frozenset([kwc1, kwc3])
        fs6 = frozenset([kwc2, kwc3])
        fs7 = frozenset([kwc1, kwc2, kwc3])
        result = so.get_min_query_dataset_distance()
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result.get(fs1), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs2), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs3), 4.24, delta=0.01)
        self.assertAlmostEqual(result.get(fs4), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs5), 1.41, delta=0.01)
        self.assertAlmostEqual(result.get(fs6), 2.83, delta=0.01)
        self.assertAlmostEqual(result.get(fs7), 1.41, delta=0.01)

    def test_get_max_keyword_similarity(self):
        query_keywords = ['family', 'food', 'outdoor']
        kwc1_keywords = ['family', 'food', 'outdoor']
        kwc2_keywords = ['food', 'family']
        kwc3_keywords = ['outdoor']
        query = KeywordCoordinate(0, 0, query_keywords)
        kwc1 = KeywordCoordinate(1, 1, kwc1_keywords)
        kwc2 = KeywordCoordinate(2, 2, kwc2_keywords)
        kwc3 = KeywordCoordinate(3, 3, kwc3_keywords)
        data = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf, normalize=False)
        fs1 = frozenset([kwc1])
        fs2 = frozenset([kwc2])
        fs3 = frozenset([kwc3])
        fs4 = frozenset([kwc1, kwc2])
        fs5 = frozenset([kwc1, kwc3])
        fs6 = frozenset([kwc2, kwc3])
        fs7 = frozenset([kwc1, kwc2, kwc3])
        result = so.get_max_keyword_similarity()
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result.get(fs1), 0.0, delta=0.01)
        self.assertAlmostEqual(result.get(fs2), 0.18, delta=0.01)
        self.assertAlmostEqual(result.get(fs3), 0.42, delta=0.01)
        self.assertAlmostEqual(result.get(fs4), 0.18, delta=0.01)
        self.assertAlmostEqual(result.get(fs5), 0.42, delta=0.01)
        self.assertAlmostEqual(result.get(fs6), 0.42, delta=0.01)
        self.assertAlmostEqual(result.get(fs7), 0.42, delta=0.01)
