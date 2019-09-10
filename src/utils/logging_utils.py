import typing

from model.keyword_coordinate import KeywordCoordinate
from utils.types import dataset_type


def dataset_comprehension(dataset: dataset_type) -> str:
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
    result = '['
    for tuple in solution_list:
        result += '(result: {}, solver: {}), '.format(result_list_comprehension(tuple[0]), tuple[1].__str__())
    result = result[:-2]
    result += ']'
    return result


def result_list_comprehension(result_list: typing.List[typing.Tuple[float, typing.Set[KeywordCoordinate]]]) -> str:
    result = '['
    for element in result_list:
        result += '({}, {}), '.format(element[0], dataset_comprehension(element[1]))
    result = result[:-2]
    result += ']'
    return result


def sets_of_set_comprehension(dataset: typing.Set[typing.Set[KeywordCoordinate]]) -> str:
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
    result = '['
    for element in list:
        result += element.__str__()
        result += ', '
    result = result[:-2]
    result += ']'
    return result
