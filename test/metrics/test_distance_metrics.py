from unittest import TestCase

import metrics.distance_metrics as mt


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
