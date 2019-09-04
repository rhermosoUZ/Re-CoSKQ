from unittest import TestCase
from costfunctions.costfunction import CostFunction
from metrics.distance_metrics import euclidean_distance, manhattan_distance
from metrics.similarity_metrics import keyword_distance
from metrics.types import dataset_type
from model.keyword_coordinate import KeywordCoordinate


class TestCostFunction(TestCase):
    def test_instantiation(self):
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        self.assertEqual(euclidean_distance.__get__, cf.distance_metric.__get__)
        self.assertEqual(keyword_distance.__get__, cf.similarity_metric.__get__)
        self.assertAlmostEqual(cf.alpha, 0.3, delta=0.01)
        self.assertAlmostEqual(cf.beta, 0.3, delta=0.01)
        self.assertAlmostEqual(cf.omega, 0.4, delta=0.01)

    def test_get_maximum_for_dataset1(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_dataset(dataset)
        self.assertAlmostEqual(result, 7.07, delta=0.01)

    def test_get_maximum_for_dataset2(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_dataset(dataset)
        self.assertAlmostEqual(result, 10.0, delta=0.01)

    def test_get_maximum_for_dataset3(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(6, 6, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_dataset(dataset)
        self.assertAlmostEqual(result, 41.01, delta=0.01)

    def test_get_maximum_for_dataset4(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(6, 6, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_dataset(dataset)
        self.assertAlmostEqual(result, 58.0, delta=0.01)

    def test_get_minimum_for_dataset1(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(6, 6, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(7, 7, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(10, 10, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_dataset(dataset)
        self.assertAlmostEqual(result, 1.41, delta=0.01)

    def test_get_minimum_for_dataset2(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(6, 6, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(7, 7, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(10, 10, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_dataset(dataset)
        self.assertAlmostEqual(result, 2.0, delta=0.01)

    def test_get_minimum_for_dataset3(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(20, 20, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(800, 800, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(9000, 9000, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(10000, 10000, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_dataset(dataset)
        self.assertAlmostEqual(result, 9.9, delta=0.01)

    def test_get_minimum_for_dataset4(self):
        keywords_dont_matter_here = ['']
        kwc1 = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(20, 20, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(800, 800, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(9000, 9000, keywords_dont_matter_here)
        kwc6 = KeywordCoordinate(10000, 10000, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5, kwc6]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_dataset(dataset)
        self.assertAlmostEqual(result, 14.0, delta=0.01)

    def test_get_maximum_for_query1(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_query(query, dataset)
        self.assertAlmostEqual(result, 7.07, delta=0.01)

    def test_get_maximum_for_query2(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_query(query, dataset)
        self.assertAlmostEqual(result, 10.0, delta=0.01)

    def test_get_maximum_for_query3(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_query(query, dataset)
        self.assertAlmostEqual(result, 49.5, delta=0.01)

    def test_get_maximum_for_query4(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_for_query(query, dataset)
        self.assertAlmostEqual(result, 70.0, delta=0.01)

    def test_get_minimum_for_query1(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_query(query, dataset)
        self.assertAlmostEqual(result, 1.41, delta=0.01)

    def test_get_minimum_for_query2(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(1, 1, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(2, 2, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(3, 3, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(4, 4, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(5, 5, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_query(query, dataset)
        self.assertAlmostEqual(result, 2.0, delta=0.01)

    def test_get_minimum_for_query3(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_query(query, dataset)
        self.assertAlmostEqual(result, 11.31, delta=0.01)

    def test_get_minimum_for_query4(self):
        keywords_dont_matter_here = ['']
        query = KeywordCoordinate(0, 0, keywords_dont_matter_here)
        kwc1 = KeywordCoordinate(8, 8, keywords_dont_matter_here)
        kwc2 = KeywordCoordinate(9, 9, keywords_dont_matter_here)
        kwc3 = KeywordCoordinate(13, 13, keywords_dont_matter_here)
        kwc4 = KeywordCoordinate(24, 24, keywords_dont_matter_here)
        kwc5 = KeywordCoordinate(35, 35, keywords_dont_matter_here)
        dataset: dataset_type = [kwc1, kwc2, kwc3, kwc4, kwc5]
        cf = CostFunction(manhattan_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_minimum_for_query(query, dataset)
        self.assertAlmostEqual(result, 16.0, delta=0.01)

    def test_get_maximum_keyword_distance1(self):
        keywords_query = ['food', 'fun', 'outdoor']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['food', 'outdoor']
        coordinates_dont_matter_here = 0
        query = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_query)
        kwc1 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc1)
        kwc2 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc2)
        kwc3 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc3)
        dataset: dataset_type = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_keyword_distance(query, dataset)
        self.assertAlmostEqual(result, 0.18, delta=0.01)

    def test_get_maximum_keyword_distance2(self):
        keywords_query = ['food', 'fun', 'outdoor']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['outdoor']
        coordinates_dont_matter_here = 0
        query = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_query)
        kwc1 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc1)
        kwc2 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc2)
        kwc3 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc3)
        dataset: dataset_type = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_keyword_distance(query, dataset)
        self.assertAlmostEqual(result, 0.42, delta=0.01)

    def test_get_maximum_keyword_distance3(self):
        keywords_query = ['food', 'fun', 'outdoor']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun']
        keywords_kwc3 = ['indoor']
        coordinates_dont_matter_here = 0
        query = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_query)
        kwc1 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc1)
        kwc2 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc2)
        kwc3 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc3)
        dataset: dataset_type = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_keyword_distance(query, dataset)
        self.assertAlmostEqual(result, 1.0, delta=0.01)

    def test_get_maximum_keyword_distance4(self):
        keywords_query = ['food', 'fun', 'outdoor']
        keywords_kwc1 = ['food', 'fun', 'outdoor']
        keywords_kwc2 = ['food', 'fun', 'outdoor']
        keywords_kwc3 = ['food', 'fun', 'outdoor']
        coordinates_dont_matter_here = 0
        query = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_query)
        kwc1 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc1)
        kwc2 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc2)
        kwc3 = KeywordCoordinate(coordinates_dont_matter_here, coordinates_dont_matter_here, keywords_kwc3)
        dataset: dataset_type = [kwc1, kwc2, kwc3]
        cf = CostFunction(euclidean_distance, keyword_distance, 0.3, 0.3, 0.4)
        result = cf.get_maximum_keyword_distance(query, dataset)
        self.assertAlmostEqual(result, 0.0, delta=0.01)
