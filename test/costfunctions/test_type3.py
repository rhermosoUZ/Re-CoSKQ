from unittest import TestCase

from costfunctions.type3 import Type3
from metrics.distance_metrics import euclidean_distance, manhattan_distance
from metrics.similarity_metrics import keyword_distance
from model.keyword_coordinate import KeywordCoordinate


class TestType3(TestCase):
    def test_instantiation(self):
        t3 = Type3(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        self.assertEqual(euclidean_distance.__get__, t3.distance_metric.__get__)
        self.assertEqual(keyword_distance.__get__, t3.similarity_metric.__get__)
        self.assertAlmostEqual(t3.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(t3.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(t3.omega, 0.4, delta=0.01)

    def test_solve1(self):
        t3 = Type3(euclidean_distance, keyword_distance, 1, 0, 0)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 1.41, delta=0.01)

    def test_solve2(self):
        t3 = Type3(euclidean_distance, keyword_distance, 0, 1, 0)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 5.66, delta=0.01)

    def test_solve3(self):
        t3 = Type3(euclidean_distance, keyword_distance, 0, 0, 1)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_solve4(self):
        t3 = Type3(manhattan_distance, keyword_distance, 1, 0, 0)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 2.0, delta=0.01)

    def test_solve5(self):
        t3 = Type3(manhattan_distance, keyword_distance, 0, 1, 0)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 8.0, delta=0.01)

    def test_solve6(self):
        t3 = Type3(manhattan_distance, keyword_distance, 0, 0, 1)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_solve7(self):
        t3 = Type3(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 2.32, delta=0.01)

    def test_solve8(self):
        t3 = Type3(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        keywords_query = ['food', 'fun', 'outdoor', 'family']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food']
        query = KeywordCoordinate(0, 0, keywords_query)
        kwc1 = KeywordCoordinate(1, 1, keywords_kwc1)
        kwc2 = KeywordCoordinate(2, 2, keywords_kwc2)
        kwc3 = KeywordCoordinate(3, 3, keywords_kwc3)
        kwc4 = KeywordCoordinate(4, 4, keywords_kwc3)
        kwc5 = KeywordCoordinate(5, 5, keywords_kwc3)
        data = [kwc1, kwc2, kwc3, kwc4, kwc5]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 3.2, delta=0.01)
