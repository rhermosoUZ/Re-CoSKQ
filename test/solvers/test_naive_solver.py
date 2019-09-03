from unittest import TestCase
from solvers.naive_solver import NaiveSolver
from costfunctions.type1 import Type1
from metrics.distance_metrics import euclidean_distance
from metrics.similarity_metrics import keyword_distance
from model.keyword_coordinate import KeywordCoordinate


class TestNaiveSolver(TestCase):
    def test_solve(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        poi1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        poi2 = KeywordCoordinate(3, 3, ['food'])
        poi3 = KeywordCoordinate(2, 2, ['outdoor'])
        poiset = [poi1, poi2, poi3]
        cf = Type1(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        ns = NaiveSolver(query, poiset, cf)
        result = ns.solve()
        self.assertAlmostEqual(result[0], 0.42, delta=0.01)
