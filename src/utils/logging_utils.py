import typing

from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.typing_definitions import dataset_type, solution_list


def dataset_comprehension(dataset: dataset_type) -> str:
    """
    Unrolls a dataset for easily comprehensible log entries.
    :param dataset: The dataset
    :return: A string with the unrolled dataset information
    """
    result = '['
    for kwc in dataset:
        result += '({}, {}) ['.format(kwc.coordinates.x, kwc.coordinates.y)
        for keyword in kwc.keywords:
            result += '\'{}\', '.format(keyword)
        result = result[:-2]
        result += '], '
    result = result[:-2]
    result += ']'
    return result


def solution_list_comprehension(solution_list: typing.List[typing.Tuple]) -> str:
    """
    Unrolls a solution list for easily comprehensible log entries.
    :param solution_list: The solution list
    :return: A string with the unrolled solution list information
    """
    result = '['
    for tuple in solution_list:
        result += '(result: {}, solver: {}), '.format(result_list_comprehension(tuple[0]), tuple[1].__str__())
    result = result[:-2]
    result += ']'
    return result


def result_list_comprehension(result_list: solution_list) -> str:
    """
    Unrolls a result list for easily comprehensible log entries.
    :param result_list: The result list
    :return: A string with the unrolled result list information
    """
    if len(result_list) == 0:
        return '[]'
    result = '['
    for element in result_list:
        result += '({}, {}), '.format(element[0], dataset_comprehension(element[1]))
    result = result[:-2]
    result += ']'
    return result


def timing_list_comprehension(timings: typing.List[typing.Tuple]) -> str:
    result = '['
    for element in timings:
        result += '({}, {}), '.format(element[0], element[1])
    result = result[:-2]
    result += ']'
    return result


def sets_of_set_comprehension(dataset: typing.Set[typing.Set[KeywordCoordinate]]) -> str:
    """
    Unrolls sets of a set for easily comprehensible log entries.
    :param dataset: The set with sets
    :return: A string with the unrolled set of sets information
    """
    result = '('
    for set in dataset:
        for kwc in set:
            result += '('
            result += kwc.__str__()
            result += '), '
    result = result[:-2]
    result += ')'
    return result


def list_comprehension(list: typing.List) -> str:
    """
    Unrolls a list for easily comprehensible log entries.
    :param list: The list
    :return: A string with the unrolled list information
    """
    result = '['
    for element in list:
        result += element.__str__()
        result += ', '
    result = result[:-2]
    result += ']'
    return result
