import os
from unittest import TestCase

from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.data_handler import write_pickle, load_pickle


class TestDataHandler(TestCase):
    def test_write_and_read_data(self):
        kwc1 = KeywordCoordinate(1, 1, ['1'])
        kwc2 = KeywordCoordinate(2, 2, ['2'])
        kwc3 = KeywordCoordinate(3, 3, ['3'])
        data = [kwc1, kwc2, kwc3]
        file_name = 'test/test.pickle'
        write_pickle(data, file_name, True)
        loaded_result = load_pickle(file_name)
        self.assertEqual(len(loaded_result), 3)
        for index in range(len(loaded_result)):
            self.assertAlmostEqual(loaded_result[index].coordinates.x, data[index].coordinates.x)
            self.assertAlmostEqual(loaded_result[index].coordinates.y, data[index].coordinates.y)
            self.assertListEqual(loaded_result[index].keywords, data[index].keywords)
        os.remove(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../' + file_name))
