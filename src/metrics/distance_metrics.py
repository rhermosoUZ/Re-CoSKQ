import logging
import math

from model.coordinate import Coordinate


def euclidean_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    logger = logging.getLogger(__name__ + '.euclidean_distance')
    logger.debug('calculating for {} and {}'.format(coordinate1, coordinate2))
    solution = math.sqrt((coordinate1.x - coordinate2.x) ** 2 + (coordinate1.y - coordinate2.y) ** 2)
    logger.debug('calculated {}'.format(solution))
    return solution


def manhattan_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    logger = logging.getLogger(__name__ + '.manhattan_distance')
    logger.debug('calculating for {} and {}'.format(coordinate1, coordinate2))
    solution = abs(coordinate1.x - coordinate2.x) + abs(coordinate1.y - coordinate2.y)
    logger.debug('calculated {}'.format(solution))
    return solution
