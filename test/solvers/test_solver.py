from unittest import TestCase

from costfunctions.costfunction import CostFunction
from solvers.solver import Solver
from model.keyword_coordinate import KeywordCoordinate
from metrics.distance_metrics import euclidean_distance
from metrics.similarity_metrics import keyword_distance


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
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        so = Solver(query, data, cf)
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
        self.assertEqual(keyword_distance.__get__, so.cost_function.similarity_metric.__get__)
        self.assertAlmostEqual(so.cost_function.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(so.cost_function.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(so.cost_function.omega, 0.4, delta=0.01)
