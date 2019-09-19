import os
from unittest import TestCase

from src.utils.data_generator import DataGenerator


class TestDataGenerator(TestCase):
    def test_instantiation(self):
        possible_keywords = ['1', '2', '3', '4', '5']
        keywords_min = 1
        keywords_max = 3
        physical_min_x = 0.0
        physical_max_x = 50.0
        physical_min_y = 1.0
        physical_max_y = 40.0
        dg = DataGenerator(possible_keywords=possible_keywords, keywords_min=keywords_min, keywords_max=keywords_max, physical_min_x=physical_min_x, physical_max_x=physical_max_x, physical_min_y=physical_min_y, physical_max_y=physical_max_y)
        self.assertListEqual(dg.possible_keywords, possible_keywords)
        self.assertEqual(dg.keywords_min, keywords_min)
        self.assertEqual(dg.keywords_max, keywords_max)
        self.assertAlmostEqual(dg.physical_min_x, physical_min_x)
        self.assertAlmostEqual(dg.physical_max_x, physical_max_x)
        self.assertAlmostEqual(dg.physical_min_y, physical_min_y)
        self.assertAlmostEqual(dg.physical_max_y, physical_max_y)

    def test_value_range(self):
        possible_keywords = ['1', '2', '3', '4', '5']
        keywords_min = 1
        keywords_max = 4
        physical_min_x = 0.0
        physical_max_x = 50.0
        physical_min_y = 0.0
        physical_max_y = 50.0
        dg = DataGenerator(possible_keywords=possible_keywords, keywords_min=keywords_min, keywords_max=keywords_max, physical_min_x=physical_min_x, physical_max_x=physical_max_x, physical_min_y=physical_min_y, physical_max_y=physical_max_y)
        result_length = 5
        result = dg.generate(result_length)
        self.assertEqual(len(result), result_length)
        for kwc in result:
            self.assertGreaterEqual(kwc.coordinates.x, physical_min_x)
            self.assertLessEqual(kwc.coordinates.x, physical_max_x)
            self.assertGreaterEqual(kwc.coordinates.y, physical_min_y)
            self.assertLessEqual(kwc.coordinates.y, physical_max_y)
            self.assertGreaterEqual(len(kwc.keywords), keywords_min)
            self.assertLessEqual(len(kwc.keywords), keywords_max)
            for kw in kwc.keywords:
                self.assertIn(kw, possible_keywords)

    def test_keyword_min_greater_length_possible_keywords(self):
        possible_keywords = ['1', '2', '3', '4', '5']
        keywords_min = 6
        keywords_max = 10
        physical_min_x = 0.0
        physical_max_x = 50.0
        physical_min_y = 0.0
        physical_max_y = 50.0
        dg = DataGenerator(possible_keywords=possible_keywords, keywords_min=keywords_min, keywords_max=keywords_max, physical_min_x=physical_min_x, physical_max_x=physical_max_x, physical_min_y=physical_min_y, physical_max_y=physical_max_y)
        result_length = 5
        result = dg.generate(result_length)
        self.assertEqual(len(result), result_length)
        for kwc in result:
            self.assertEqual(len(kwc.keywords), len(possible_keywords))

    def test_pickle(self):
        possible_keywords = ['1', '2', '3', '4', '5']
        keywords_min = 6
        keywords_max = 10
        physical_min_x = 0.0
        physical_max_x = 50.0
        physical_min_y = 0.0
        physical_max_y = 50.0
        dg = DataGenerator(possible_keywords=possible_keywords, keywords_min=keywords_min, keywords_max=keywords_max, physical_min_x=physical_min_x, physical_max_x=physical_max_x, physical_min_y=physical_min_y, physical_max_y=physical_max_y)
        result_length = 5
        result = dg.generate(result_length)
        file_name = 'test/test.pickle'
        dg.write_pickle(result, file_name, True)
        loaded_result = dg.load_pickle(file_name)
        self.assertEqual(len(loaded_result), result_length)
        for index in range(len(loaded_result)):
            self.assertAlmostEqual(loaded_result[index].coordinates.x, result[index].coordinates.x)
            self.assertAlmostEqual(loaded_result[index].coordinates.y, result[index].coordinates.y)
            self.assertListEqual(loaded_result[index].keywords, result[index].keywords)
        generated_result = dg.generate_pickle(result_length, file_name, True)
        loaded_result2 = dg.load_pickle(file_name)
        self.assertEqual(len(loaded_result2), result_length)
        for index in range(len(loaded_result2)):
            self.assertAlmostEqual(loaded_result2[index].coordinates.x, generated_result[index].coordinates.x)
            self.assertAlmostEqual(loaded_result2[index].coordinates.y, generated_result[index].coordinates.y)
            self.assertListEqual(loaded_result2[index].keywords, generated_result[index].keywords)
        os.remove(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../' + file_name))
