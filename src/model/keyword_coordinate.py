from __future__ import annotations

from model.coordinate import Coordinate
import typing


class KeywordCoordinate:
    def __init__(self, x: float, y: float, keyword_set: typing.List[str]):
        self.coordinates: Coordinate = Coordinate(x, y)
        self.keywords: typing.List[str] = keyword_set

    def __str__(self):
        return '({}, {}), {}'.format(self.coordinates.x, self.coordinates.y, self.keywords)
