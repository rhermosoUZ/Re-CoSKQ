from unittest import TestCase

import metrics.similarity_metrics as mt
from model.keyword_coordinate import KeywordCoordinate
from utils.types import sim_dataset_type, sim_tuple_type


class TestSimilarityMetrics(TestCase):
    def test_cosine_similarity1(self):
        v1 = [0, 1, 0, 1]
        v2 = [0, 0, 1, 1]
        result = mt.cosine_similarity(v1, v2)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_cosine_similarity2(self):
        v1 = [0, 0, 0, 0]
        v2 = [1, 1, 1, 1]
        self.assertRaises(ValueError, mt.cosine_similarity, v1, v2)

    def test_cosine_similarity3(self):
        v1 = [1, 1, 1, 1]
        v2 = [1, 1, 1, 1]
        result = mt.cosine_similarity(v1, v2)
        self.assertAlmostEqual(result, 1.0, delta=0.01)

    def test_cosine_similarity4(self):
        v1 = [0, 0, 0, 0]
        v2 = [0, 0, 0, 0]
        self.assertRaises(ValueError, mt.cosine_similarity, v1, v2)

    def test_cosine_similarity5(self):
        v1 = [1, 1, 1, 0]
        v2 = [1, 1, 1, 1]
        result = mt.cosine_similarity(v1, v2)
        self.assertAlmostEqual(result, 0.87, delta=0.01)

    def test_cosine_similarity6(self):
        v1 = [1, 0, 0, 0]
        v2 = [1, 1, 1, 1]
        result = mt.cosine_similarity(v1, v2)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_one_hot_encode1(self):
        kw_list1 = ['1', '2']
        kw_list2 = ['2', '3']
        kw_list_combined = ['1', '2', '3', '4']
        result1, result2 = mt.one_hot_encode(kw_list1, kw_list2, kw_list_combined)
        self.assertListEqual(result1, [1, 1, 0, 0])
        self.assertListEqual(result2, [0, 1, 1, 0])

    def test_one_hot_encode2(self):
        kw_list1 = ['1', '2']
        kw_list2 = ['4']
        kw_list_combined = ['1', '2', '3', '4', '5']
        result1, result2 = mt.one_hot_encode(kw_list1, kw_list2, kw_list_combined)
        self.assertListEqual(result1, [1, 1, 0, 0, 0])
        self.assertListEqual(result2, [0, 0, 0, 1, 0])

    def test_create_keyword_vector1(self):
        kwv1 = ['kw1', 'kw2', 'kw3']
        kwv2 = ['kw4', 'kw5', 'kw6']
        result = mt.create_keyword_vector(kwv1, kwv2)
        self.assertEqual(self.get_sim_counter(result), 0)
        self.assertEqual(sum(result[0]), 3)
        self.assertEqual(len(result[0]), 6)
        self.assertEqual(sum(result[1]), 3)
        self.assertEqual(len(result[1]), 6)

    def test_create_keyword_vector2(self):
        kwv1 = ['kw1', 'kw2', 'kw3']
        kwv2 = ['kw1', 'kw2', 'kw3']
        result = mt.create_keyword_vector(kwv1, kwv2)
        self.assertEqual(self.get_sim_counter(result), 3)
        self.assertEqual(sum(result[0]), 3)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(sum(result[1]), 3)
        self.assertEqual(len(result[1]), 3)

    def test_create_keyword_vector3(self):
        kwv1 = ['kw1', 'kw2', 'kw3']
        kwv2 = ['kw4', 'kw2', 'kw3']
        result = mt.create_keyword_vector(kwv1, kwv2)
        self.assertEqual(self.get_sim_counter(result), 2)
        self.assertEqual(sum(result[0]), 3)
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(sum(result[1]), 3)
        self.assertEqual(len(result[1]), 4)

    def test_create_combined_keyword_vector1(self):
        kwv1 = ['kw1', 'kw2', 'kw3']
        kwv2 = ['kw4', 'kw2', 'kw3']
        kwc1 = KeywordCoordinate(0, 0, kwv1)
        kwc2 = KeywordCoordinate(0, 0, kwv2)
        kwc3 = KeywordCoordinate(0, 0, kwv2)
        kwc_list = [kwc2, kwc3]
        result = mt.create_combined_keyword_vector(kwc1, kwc_list)
        combined_list = kwv1 + kwv2
        for element in result:
            self.assertTrue(element in combined_list)
        self.assertEqual(len(result), 4)

    def test_create_combined_keyword_vector2(self):
        kwv1 = ['kw1', 'kw2', 'kw3']
        kwv2 = ['kw4', 'kw2', 'kw5']
        kwv3 = ['kw6', 'kw7', 'kw8']
        kwc1 = KeywordCoordinate(0, 0, kwv1)
        kwc2 = KeywordCoordinate(0, 0, kwv2)
        kwc3 = KeywordCoordinate(0, 0, kwv3)
        kwc_list = [kwc2, kwc3]
        result = mt.create_combined_keyword_vector(kwc1, kwc_list)
        combined_list = kwv1 + kwv2 + kwv3
        for element in result:
            self.assertTrue(element in combined_list)
        self.assertEqual(len(result), 8)

    def test_separated_cosine_similarity1(self):
        kwl1 = ['kw1', 'kw2', 'kw3', 'kw4']
        kwl2 = ['kw1', 'kw2', 'kw3', 'kw4']
        result = mt.separated_cosine_similarity(kwl1, kwl2)
        self.assertAlmostEqual(result, 0.0, delta=0.01)

    def test_separated_cosine_similarity2(self):
        kwl1 = ['kw1', 'kw2']
        kwl2 = ['kw3', 'kw4']
        result = mt.separated_cosine_similarity(kwl1, kwl2)
        self.assertAlmostEqual(result, 1.0, delta=0.01)

    def test_separated_cosine_similarity3(self):
        kwl1 = ['kw1', 'kw2', 'kw3']
        kwl2 = ['kw3', 'kw4']
        result = mt.separated_cosine_similarity(kwl1, kwl2)
        self.assertAlmostEqual(result, 0.59, delta=0.01)

    def test_separated_cosine_similarity4(self):
        kwl1 = ['kw1', 'kw2', 'kw3']
        kwl2 = ['kw2', 'kw3', 'kw4']
        result = mt.separated_cosine_similarity(kwl1, kwl2)
        self.assertAlmostEqual(result, 0.33, delta=0.01)

    def test_separated_cosine_similarity5(self):
        kwl1 = ['kw1', 'kw2', 'kw3', 'kw4']
        kwl2 = ['kw2', 'kw3', 'kw4']
        result = mt.separated_cosine_similarity(kwl1, kwl2)
        self.assertAlmostEqual(result, 0.13, delta=0.01)

    def test_combined_cosine_similarity1(self):
        kw_list1 = ['1', '2']
        kw_list2 = ['2', '3']
        combined_kw_list = ['1', '2', '3']
        result = mt.combined_cosine_similarity(kw_list1, kw_list2, combined_kw_list)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_combined_cosine_similarity2(self):
        kw_list1 = ['1', '2']
        kw_list2 = ['2', '3']
        combined_kw_list = ['1', '2', '3', '4']
        result = mt.combined_cosine_similarity(kw_list1, kw_list2, combined_kw_list)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_combined_cosine_similarity3(self):
        kw_list1 = ['1', '2', '3']
        kw_list2 = ['2', '3']
        combined_kw_list = ['1', '2', '3', '4']
        result = mt.combined_cosine_similarity(kw_list1, kw_list2, combined_kw_list)
        self.assertAlmostEqual(result, 0.18, delta=0.01)


    def test_combined_cosine_similarity4(self):
        kw_list1 = ['1', '2', '3']
        kw_list2 = ['2', '3']
        combined_kw_list = ['1', '2', '3', '4', '5']
        result = mt.combined_cosine_similarity(kw_list1, kw_list2, combined_kw_list)
        self.assertAlmostEqual(result, 0.18, delta=0.01)

    def test_find_subsets0(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 0)
        self.assertEqual(len(subsets), 1)
        for subset in subsets:
            self.assertEqual(len(subset), 0)

    def test_find_subsets1(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 1)
        self.assertEqual(len(subsets), 4)
        for subset in subsets:
            self.assertEqual(len(subset), 1)

    def test_find_subsets2(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 2)
        self.assertEqual(len(subsets), 6)
        for subset in subsets:
            self.assertEqual(len(subset), 2)

    def test_find_subsets3(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 3)
        self.assertEqual(len(subsets), 4)
        for subset in subsets:
            self.assertEqual(len(subset), 3)

    def test_find_subsets4(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 4)
        self.assertEqual(len(subsets), 1)
        for subset in subsets:
            self.assertEqual(len(subset), 4)

    def test_find_subsets5(self):
        kwc1 = KeywordCoordinate(0, 0, ['0'])
        kwc2 = KeywordCoordinate(1, 1, ['1'])
        kwc3 = KeywordCoordinate(2, 2, ['2'])
        kwc4 = KeywordCoordinate(3, 3, ['3'])
        superset = [kwc1, kwc2, kwc3, kwc4]
        subsets = mt.find_subsets(superset, 5)
        print(subsets)
        self.assertEqual(len(subsets), 1)
        for subset in subsets:
            self.assertEqual(len(subset), 0)

    def get_sim_counter(self, result_tuple: sim_tuple_type) -> int:
        result_vector1: sim_dataset_type = result_tuple[0]
        result_vector2: sim_dataset_type = result_tuple[1]
        sim_counter = 0
        for index in range(len(result_vector1)):
            if result_vector1[index] == result_vector2[index]:
                sim_counter += 1
        return sim_counter

    def test_get_sim_counter1(self):
        v1 = [0, 0, 1]
        v2 = [0, 1, 1]
        rt = (v1, v2)
        self.assertEqual(self.get_sim_counter(rt), 2)

    def test_get_sim_counter2(self):
        v1 = [0, 1, 0]
        v2 = [1, 0, 1]
        rt = (v1, v2)
        self.assertEqual(self.get_sim_counter(rt), 0)

    def test_get_sim_counter3(self):
        v1 = [1, 1, 1]
        v2 = [1, 1, 1]
        rt = (v1, v2)
        self.assertEqual(self.get_sim_counter(rt), 3)
