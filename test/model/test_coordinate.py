from unittest import TestCase
from model.coordinate import Coordinate


class TestCoordinate(TestCase):
    def test_instantiation(self):
        x = 5
        y = 7
        c = Coordinate(x, y)
        self.assertEqual(c.x, x)
        self.assertEqual(c.y, y)
