from unittest import TestCase

import metrics.distance_metrics as mt
from model.keyword_coordinate import KeywordCoordinate


class TestDistanceMetrics(TestCase):
    def test_euclidean_distance(self):
        x1 = 3
        y1 = 4
        x2 = 7
        y2 = 2
        c1 = mt.Coordinate(x1, y1)
        c2 = mt.Coordinate(x2, y2)
        result = mt.euclidean_distance(c1, c2)
        self.assertAlmostEqual(result, 4.47, delta=0.01)

    def test_manhattan_distance(self):
        x1 = 3
        y1 = 4
        x2 = 7
        y2 = 2
        c1 = mt.Coordinate(x1, y1)
        c2 = mt.Coordinate(x2, y2)
        result = mt.manhattan_distance(c1, c2)
        self.assertEqual(result, 6)

    def test_normalize_data(self):
        query = KeywordCoordinate(2, 1, ['family', 'food', 'outdoor'])
        kwc1 = KeywordCoordinate(0, 0, ['family'])
        kwc2 = KeywordCoordinate(3, 2, ['food'])
        kwc3 = KeywordCoordinate(1, 5, ['outdoor'])
        data = [kwc1, kwc2, kwc3]
        norm_query, norm_data, max_x, min_x, max_y, min_y = mt.normalize_data(query, data)
        self.assertAlmostEqual(norm_query.coordinates.x, 0.66, delta=0.01)
        self.assertAlmostEqual(norm_query.coordinates.y, 0.20, delta=0.01)
        self.assertAlmostEqual(norm_data[0].coordinates.x, 0.0, delta=0.01)
        self.assertAlmostEqual(norm_data[0].coordinates.y, 0.0, delta=0.01)
        self.assertAlmostEqual(norm_data[1].coordinates.x, 1.0, delta=0.01)
        self.assertAlmostEqual(norm_data[1].coordinates.y, 0.4, delta=0.01)
        self.assertAlmostEqual(norm_data[2].coordinates.x, 0.33, delta=0.01)
        self.assertAlmostEqual(norm_data[2].coordinates.y, 1.0, delta=0.01)
        self.assertEqual(max_x, 3)
        self.assertEqual(min_x, 0)
        self.assertEqual(max_y, 5)
        self.assertEqual(min_y, 0)

    def test_denormalize(self):
        cost_doesnt_matter = 0.0
        kwc1 = KeywordCoordinate(0.0, 0.0, ['family'])
        kwc2 = KeywordCoordinate(1.0, 0.4, ['food'])
        kwc3 = KeywordCoordinate(0.33, 1.0, ['outdoor'])
        data = [(cost_doesnt_matter, [kwc1, kwc2, kwc3])]
        result = mt.denormalize_result_data(data, 3, 0, 5, 0)
        self.assertAlmostEqual(result[0][1][0].coordinates.x, 0.0, delta=0.02)
        self.assertAlmostEqual(result[0][1][0].coordinates.y, 0.0, delta=0.02)
        self.assertAlmostEqual(result[0][1][1].coordinates.x, 3.0, delta=0.02)
        self.assertAlmostEqual(result[0][1][1].coordinates.y, 2.0, delta=0.02)
        self.assertAlmostEqual(result[0][1][2].coordinates.x, 1.0, delta=0.02)
        self.assertAlmostEqual(result[0][1][2].coordinates.y, 5.0, delta=0.02)
