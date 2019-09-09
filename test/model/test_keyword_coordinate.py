from unittest import TestCase

from model.keyword_coordinate import KeywordCoordinate


class TestKeywordCoordinate(TestCase):
    def test_instantiation(self):
        x = 3
        y = 8
        kw = ['keyword 1', 'kw2', '3']
        kwc = KeywordCoordinate(x, y, kw)
        self.assertEqual(kwc.coordinates.x, x)
        self.assertEqual(kwc.coordinates.y, y)
        self.assertEqual(kwc.keywords, kw)
