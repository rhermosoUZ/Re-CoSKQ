from __future__ import annotations

import math, typing
import itertools
from metrics.types import sim_dataset_type, keyword_dataset_type, dataset_type

# Cosine Similarity
from model.keyword_coordinate import KeywordCoordinate


def cosine_similarity(dataset1: sim_dataset_type, dataset2: sim_dataset_type) -> float:
    if len(dataset1) != len(dataset2):
        raise ValueError('Both datasets have to be of the same length.')
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
    return numerator / denominator


def create_keyword_vector(keyword_list1: keyword_dataset_type, keyword_list2: keyword_dataset_type) -> typing.Tuple[
    typing.List[int], typing.List[int]]:
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
    return (result_vector1, result_vector2)


def keyword_distance(query_keyword_list, poi_keyword_list) -> float:
    # dist(k 1 , k 2 ) = 1 − sim(k 1 , k 2 )
    kw_vectors = create_keyword_vector(query_keyword_list, poi_keyword_list)
    cosim = cosine_similarity(kw_vectors[0], kw_vectors[1])
    return 1 - cosim


# https://stackoverflow.com/questions/374626/how-can-i-find-all-the-subsets-of-a-set-with-exactly-n-elements#374645
def find_subsets(S: typing.List, m: int):
    return set(itertools.combinations(S, m))
