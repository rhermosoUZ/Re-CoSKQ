import math
from unittest import TestCase

from costfunctions.type3 import Type3
from metrics.distance_metrics import euclidean_distance, manhattan_distance
from metrics.similarity_metrics import separated_cosine_similarity
from model.keyword_coordinate import KeywordCoordinate


class TestType3(TestCase):
    def test_instantiation(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, False)
        self.assertEqual(euclidean_distance.__get__, t3.distance_metric.__get__)
        self.assertEqual(separated_cosine_similarity.__get__, t3.similarity_metric.__get__)
        self.assertAlmostEqual(t3.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(t3.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(t3.omega, 0.4, delta=0.01)
        self.assertAlmostEqual(t3.query_distance_threshold, 0.5, delta=0.01)
        self.assertAlmostEqual(t3.dataset_distance_threshold, 0.6, delta=0.01)
        self.assertAlmostEqual(t3.keyword_similarity_threshold, 0.7, delta=0.01)
        self.assertEqual(t3.disable_thresholds, False)

    def test_solve1(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 1, 0, 0, disable_thresholds=True)
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
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0, 1, 0, disable_thresholds=True)
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
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0, 0, 1, disable_thresholds=True)
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
        t3 = Type3(manhattan_distance, separated_cosine_similarity, 1, 0, 0, disable_thresholds=True)
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
        t3 = Type3(manhattan_distance, separated_cosine_similarity, 0, 1, 0, disable_thresholds=True)
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
        t3 = Type3(manhattan_distance, separated_cosine_similarity, 0, 0, 1, disable_thresholds=True)
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
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
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
        t3 = Type3(manhattan_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
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

    def test_threshold1(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, 0.2, math.inf, math.inf)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        kwc2 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 0.04, delta=0.01)

    def test_threshold2(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.3, 0.3, 0.4, 0.1, math.inf, math.inf)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        kwc2 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, math.inf, delta=0.01)

    def test_threshold3(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.0, 0.3, 0.7, math.inf, 0.2, math.inf)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        kwc2 = KeywordCoordinate(0.2, 0.2, ['keyword1', 'keyword2', 'keyword3'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 0.04, delta=0.01)

    def test_threshold4(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.0, 0.3, 0.7, math.inf, 0.1, math.inf)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0.1, 0.1, ['keyword1', 'keyword2', 'keyword3'])
        kwc2 = KeywordCoordinate(0.2, 0.2, ['keyword1', 'keyword2', 'keyword3'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, math.inf, delta=0.01)

    def test_threshold5(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.25, 0.25, 0.5, math.inf, math.inf, 0.5)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0, 0, ['keyword1'])
        kwc2 = KeywordCoordinate(0, 0, ['keyword2'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, 0.21, delta=0.01)

    def test_threshold6(self):
        t3 = Type3(euclidean_distance, separated_cosine_similarity, 0.25, 0.25, 0.5, math.inf, math.inf, 0.4)
        query = KeywordCoordinate(0, 0, ['keyword1', 'keyword2', 'keyword3'])
        kwc1 = KeywordCoordinate(0, 0, ['keyword1'])
        kwc2 = KeywordCoordinate(0, 0, ['keyword2'])
        data = [kwc1, kwc2]
        result = t3.solve(query, data)
        self.assertAlmostEqual(result, math.inf, delta=0.01)
