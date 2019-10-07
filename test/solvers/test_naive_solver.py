from unittest import TestCase

from src.costfunctions.type1 import Type1
from src.costfunctions.type3 import Type3
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import separated_cosine_similarity, combined_cosine_similarity, \
    word2vec_cosine_similarity
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_generator import DataGenerator
from src.utils.data_handler import load_word2vec_model, calculate_model_subset


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

    def test_precalculated_max_query_dataset(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food', 'family'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        cf = Type1(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf)
        pre_qd = ns.get_max_query_dataset_distance()
        result = ns.solve()
        cf.precalculated_query_dataset_dict = pre_qd
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_precalculated_min_query_dataset(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food', 'family'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        cf = Type3(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf)
        pre_qd = ns.get_min_query_dataset_distance()
        result = ns.solve()
        cf.precalculated_query_dataset_dict = pre_qd
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_precalculated_max_inter_dataset(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food', 'family'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        cf = Type1(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf)
        pre_id = ns.get_max_inter_dataset_distance()
        result = ns.solve()
        cf.precalculated_inter_dataset_dict = pre_id
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_precalculated_max_keyword_similarity(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food', 'family'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        cf = Type1(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf)
        pre_ks = ns.get_max_keyword_similarity()
        result = ns.solve()
        cf.precalculated_keyword_similarity_dict = pre_ks
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_precalculated_word2vec(self):
        query = KeywordCoordinate(0, 0, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(1, 1, ['family', 'food', 'outdoor'])
        kwc2 = KeywordCoordinate(3, 3, ['food', 'family'])
        kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        model = calculate_model_subset(query, data, load_word2vec_model())
        cf = Type3(euclidean_distance, word2vec_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True, model=model)
        ns = NaiveSolver(query, data, cf)
        result = ns.solve()
        pre_qd = ns.get_query_dataset_distance()
        pre_id = ns.get_inter_dataset_distance()
        pre_ks = ns.get_keyword_similarity()
        cf.precalculated_query_dataset_dict = pre_qd
        cf.precalculated_inter_dataset_dict = pre_id
        cf.precalculated_keyword_similarity_dict = pre_ks
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_complex_precalculations(self):
        query = KeywordCoordinate(5, 6, ['culture'])
        kwc1 = KeywordCoordinate(2, 1, ['family', 'rest', 'indoor'])
        kwc2 = KeywordCoordinate(0, 2, ['science', 'culture', 'history'])
        kwc3 = KeywordCoordinate(0, 0, ['food', 'outdoor', 'sports'])
        data = [kwc1, kwc2, kwc3]
        cf = Type1(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf, result_length=100)
        result = ns.solve()
        pre_qd = ns.get_query_dataset_distance()
        pre_id = ns.get_inter_dataset_distance()
        pre_ks = ns.get_keyword_similarity()
        cf.precalculated_query_dataset_dict = pre_qd
        cf.precalculated_inter_dataset_dict = pre_id
        cf.precalculated_keyword_similarity_dict = pre_ks
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_complex_generated_precalculations(self):
        possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
        dg = DataGenerator(possible_keywords)
        query: KeywordCoordinate = dg.generate(1)[0]
        data = dg.generate(10)
        cf = Type1(euclidean_distance, combined_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True)
        ns = NaiveSolver(query, data, cf, result_length=100)
        result = ns.solve()
        pre_qd = ns.get_query_dataset_distance()
        pre_id = ns.get_inter_dataset_distance()
        pre_ks = ns.get_keyword_similarity()
        cf.precalculated_query_dataset_dict = pre_qd
        cf.precalculated_inter_dataset_dict = pre_id
        cf.precalculated_keyword_similarity_dict = pre_ks
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)

    def test_complex_generated_word2vec_precalculations(self):
        possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
        dg = DataGenerator(possible_keywords)
        query: KeywordCoordinate = dg.generate(1)[0]
        data = dg.generate(5)
        model = calculate_model_subset(query, data, load_word2vec_model())
        cf = Type1(euclidean_distance, word2vec_cosine_similarity, 0.3, 0.3, 0.4, disable_thresholds=True, model=model)
        ns = NaiveSolver(query, data, cf, result_length=100)
        result = ns.solve()
        pre_qd = ns.get_query_dataset_distance()
        pre_id = ns.get_inter_dataset_distance()
        pre_ks = ns.get_keyword_similarity()
        cf.precalculated_query_dataset_dict = pre_qd
        cf.precalculated_inter_dataset_dict = pre_id
        cf.precalculated_keyword_similarity_dict = pre_ks
        result_pre = ns.solve()
        for index in range(len(result)):
            self.assertAlmostEqual(result[index][0], result_pre[index][0], delta=0.01)
            key_list = list(result[index][1])
            key_list_pre = list(result_pre[index][1])
            for list_index in range(len(key_list)):
                self.assertAlmostEqual(key_list[list_index].coordinates.x, key_list_pre[list_index].coordinates.x)
                self.assertAlmostEqual(key_list[list_index].coordinates.y, key_list_pre[list_index].coordinates.y)
                self.assertListEqual(key_list[list_index].keywords, key_list_pre[list_index].keywords)
