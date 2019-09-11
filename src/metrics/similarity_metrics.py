from __future__ import annotations

import itertools
import logging
import math
import typing

from utils.logging_utils import dataset_comprehension, sets_of_set_comprehension
from utils.types import sim_dataset_type, keyword_dataset_type, dataset_type


# Cosine Similarity


# TODO if a set of keywords covers the query plus additional categories the similarity declines. Is this wanted?
def cosine_similarity(dataset1: sim_dataset_type, dataset2: sim_dataset_type) -> float:
    logger = logging.getLogger(__name__ + '.cosine_similarity')
    logger.debug('calculating for {} and {}'.format(dataset1, dataset2))
    if len(dataset1) != len(dataset2):
        msg = 'Both datasets have to be of the same length.'
        logger.error(msg)
        raise ValueError(msg)
    if sum(dataset1) == 0 or sum(dataset2) == 0:
        msg = 'Neither dataset may only consist of 0-values.'
        logger.error(msg)
        raise ValueError(msg)
    # Numerator
    numerator = 0
    for index in range(len(dataset1)):
        numerator += (dataset1[index] * dataset2[index])
    # Denominator
    a = 0
    b = 0
    for index in range(len(dataset1)):
        a += dataset1[index] ** 2
        b += dataset2[index] ** 2
    denominator = math.sqrt(a) * math.sqrt(b)
    solution = numerator / denominator
    logger.debug('calculated {}'.format(solution))
    return solution


# TODO generate general keyword vector over the entire set of keyword coordinates
def create_keyword_vector(keyword_list1: keyword_dataset_type, keyword_list2: keyword_dataset_type) -> typing.Tuple[
    typing.List[int], typing.List[int]]:
    logger = logging.getLogger(__name__ + '.create_keyword_vector')
    logger.debug('calculating for {} and {}'.format(keyword_list1, keyword_list2))
    merged_list = list(map(str.lower, (keyword_list1 + keyword_list2)))
    vector = list(set(merged_list))
    del (merged_list)
    result_vector1 = []
    result_vector2 = []
    for element in vector:
        if element in keyword_list1:
            result_vector1.append(1)
        else:
            result_vector1.append(0)
        if element in keyword_list2:
            result_vector2.append(1)
        else:
            result_vector2.append(0)
    solution = (result_vector1, result_vector2)
    logger.debug('calculated {}'.format(solution))
    return solution


def keyword_distance(query_keyword_list, data_keyword_list) -> float:
    logger = logging.getLogger(__name__ + '.keyword_distance')
    logger.debug('calculating for query {} and dataset {}'.format(query_keyword_list, data_keyword_list))
    kw_vectors = create_keyword_vector(query_keyword_list, data_keyword_list)
    solution = 1 - cosine_similarity(kw_vectors[0], kw_vectors[1])
    logger.debug('calculated {}'.format(solution))
    return solution


# https://stackoverflow.com/questions/374626/how-can-i-find-all-the-subsets-of-a-set-with-exactly-n-elements#374645
def find_subsets(input_set: dataset_type, subset_size: int):
    logger = logging.getLogger(__name__ + '.find_subsets')
    logger.debug('finding all subsets of length {} in set {}'.format(subset_size, dataset_comprehension(input_set)))
    if subset_size > len(input_set):
        solution = set(itertools.combinations(input_set, 0))
    else:
        solution = set(itertools.combinations(input_set, subset_size))
    logger.debug('found {}'.format(sets_of_set_comprehension(solution)))
    return solution
