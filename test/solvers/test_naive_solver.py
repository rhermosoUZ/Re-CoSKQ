from unittest import TestCase

from src.costfunctions.type1 import Type1
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import separated_cosine_similarity
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.naive_solver import NaiveSolver


class TestNaiveSolver(TestCase):
    def test_solve(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        cf = Type1(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf, normalize=False, result_length=1)
        result = ns.solve()
        self.assertAlmostEqual(result[0][0], 0.42, delta=0.01)
