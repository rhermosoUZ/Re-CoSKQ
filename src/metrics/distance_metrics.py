import math

from model.coordinate import Coordinate


def euclidean_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    return math.sqrt((coordinate1.x - coordinate2.x) ** 2 + (coordinate1.y - coordinate2.y) ** 2)


def manhattan_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    return abs(coordinate1.x - coordinate2.x) + abs(coordinate1.y - coordinate2.y)
