import copy
import logging
import math
import typing

from src.model.coordinate import Coordinate
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension, result_list_comprehension
from src.utils.typing_definitions import dataset_type, solution_list


def euclidean_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    """
    Calculates the euclidean distance between two coordinates.
    :param coordinate1: The first coordinate
    :param coordinate2: The second coordinate
    :return: The euclidean distance
    """
    logger = logging.getLogger(__name__ + '.euclidean_distance')
    logger.debug('calculating for {} and {}'.format(coordinate1, coordinate2))
    solution = math.sqrt((coordinate1.x - coordinate2.x) ** 2 + (coordinate1.y - coordinate2.y) ** 2)
    logger.debug('calculated {}'.format(solution))
    return solution


def manhattan_distance(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    """
    Calculates the manhattan distance between two coordinates.
    :param coordinate1: The first coordinate
    :param coordinate2: The second coordinate
    :return: The manhattan distance
    """
    logger = logging.getLogger(__name__ + '.manhattan_distance')
    logger.debug('calculating for {} and {}'.format(coordinate1, coordinate2))
    solution = abs(coordinate1.x - coordinate2.x) + abs(coordinate1.y - coordinate2.y)
    logger.debug('calculated {}'.format(solution))
    return solution


def normalize_data(query: KeywordCoordinate, dataset: dataset_type) -> typing.Tuple[
    KeywordCoordinate, typing.List[KeywordCoordinate], float, float, float, float]:
    """
    Calculates the normalizes query, dataset and parameters to undo this normalization.
    :param query: The query
    :param dataset: The dataset
    :return: A tuple with: the normalized query, the normalized dataset, the denormalization parameter max_x,  the denormalization parameter min_x, the denormalization parameter max_y and the denormalization parameter min_y,
    """
    logger = logging.getLogger(__name__ + '.normalize_data')
    logger.debug('calculation for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
    data = copy.deepcopy(dataset)
    
    # Cambio de Ramon (20200903)
    # data.append(copy.deepcopy(query)) // Añade una lista a una lista que solo contiene KeywordCoordinates
    data.append(copy.deepcopy(query[0]))
    list_of_x = []
    list_of_y = []
    
    type(data)
    for kwc in data:  
        list_of_x.append(kwc.coordinates.x)
        list_of_y.append(kwc.coordinates.y)
    min_x = min(list_of_x)
    min_y = min(list_of_y)
    max_x = max(list_of_x)
    max_y = max(list_of_y)
    for index in range(len(data)):
        new_x = (data[index].coordinates.x - min_x) / (max_x - min_x)
        new_y = (data[index].coordinates.y - min_y) / (max_y - min_y)
        data[index].coordinates.x = new_x
        data[index].coordinates.y = new_y
    logger.debug('calculated query {} and dataset {}'.format(data[-1:][0], dataset_comprehension(data[:-1])))
    return (data[-1:][0], data[:-1], max_x, min_x, max_y, min_y)


def denormalize_result_data(result_list: solution_list, max_x: float, min_x: float, max_y: float,
                            min_y: float) -> solution_list:
    """
    Calculates the denormalized results.
    :param result_list: The normalized results
    :param max_x: Denormalization parameter max_x
    :param min_x: Denormalization parameter min_x
    :param max_y: Denormalization parameter max_y
    :param min_y: Denormalization parameter min_y
    :return: The denormalized list of results
    """
    logger = logging.getLogger(__name__ + '.denormalize_result_data')
    logger.debug('calculation for result {}, max_x {}, min_x {}, max_y {} and min_y {}'.format(result_list_comprehension(result_list), max_x, min_x, max_y, min_y))
    result: solution_list = []
    for solution_tuple in result_list:
        solution_tuple_copy = copy.deepcopy(solution_tuple)
        for kwc in solution_tuple_copy[1]:
            kwc.coordinates.x = kwc.coordinates.x * (max_x - min_x) + min_x
            kwc.coordinates.y = kwc.coordinates.y * (max_y - min_y) + min_y
        result.append(solution_tuple_copy)
    logger.debug('calculated results {}'.format(result_list_comprehension(result)))
    return result
