from __future__ import annotations

import logging
import typing

from src.model.coordinate import Coordinate


class KeywordCoordinate:
    def __init__(self, x: float, y: float, keyword_set: typing.List[str]):
        """
        Constructs a KeywordCoordinate object. This object combines a physical 2D-location with associated keywords.
        :param x: The x coordinate
        :param y: The y coordinate
        :param keyword_set: A list with all the keywords
        """
        logger = logging.getLogger(__name__)
        self.coordinates: Coordinate = Coordinate(x, y)
        self.keywords: typing.List[str] = keyword_set
        logger.debug('created at ({}, {}) with the keywords {}'.format(self.coordinates.x, self.coordinates.y, self.keywords))

    def __str__(self):
        return '({}, {}), {}'.format(self.coordinates.x, self.coordinates.y, self.keywords)
