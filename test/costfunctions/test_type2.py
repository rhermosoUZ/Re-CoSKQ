from unittest import TestCase
from costfunctions.type2 import Type2
from metrics.similarity_metrics import keyword_distance
from metrics.distance_metrics import euclidean_distance, manhattan_distance
from model.keyword_coordinate import KeywordCoordinate


class TestType2(TestCase):
    def test_instantiation(self):
        t1 = Type2(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        self.assertEqual(euclidean_distance.__get__, t1.distance_metric.__get__)
        self.assertEqual(keyword_distance.__get__, t1.similarity_metric.__get__)
        self.assertAlmostEqual(t1.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(t1.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(t1.omega, 0.4, delta=0.01)

    def test_solve1(self):
        t1 = Type2(euclidean_distance, keyword_distance, 1, 0, 0)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 7.07, delta=0.01)

    def test_solve2(self):
        t1 = Type2(euclidean_distance, keyword_distance, 0, 1, 0)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 5.66, delta=0.01)

    def test_solve3(self):
        t1 = Type2(euclidean_distance, keyword_distance, 0, 0, 1)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_solve4(self):
        t1 = Type2(manhattan_distance, keyword_distance, 1, 0, 0)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 10.0, delta=0.01)

    def test_solve5(self):
        t1 = Type2(manhattan_distance, keyword_distance, 0, 1, 0)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 8.0, delta=0.01)

    def test_solve6(self):
        t1 = Type2(manhattan_distance, keyword_distance, 0, 0, 1)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_solve7(self):
        t1 = Type2(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 2.12, delta=0.01)

    def test_solve8(self):
        t1 = Type2(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
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
        result = t1.solve(query, data)
        self.assertAlmostEqual(result, 3.0, delta=0.01)