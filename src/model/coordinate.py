import logging


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        logging.getLogger(__name__).debug('created at ({}, {})'.format(self.x, self.y))

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)
