import logging


class Coordinate:
    def __init__(self, x: float, y: float):
        """
        Constructs a Coordinate object. This class keeps track of the physical 2D-location of KeywordCoordinates.
        :param x: The x coordinate
        :param y: The y coordinate
        """
        self.x = x
        self.y = y
        logging.getLogger(__name__).debug('created at ({}, {})'.format(self.x, self.y))

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)
