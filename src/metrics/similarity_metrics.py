from __future__ import annotations

import itertools
import logging
import math
import typing

import numpy as np
import pandas as pd

from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension, sets_of_set_comprehension
from src.utils.typing_definitions import sim_dataset_type, keyword_dataset_type, dataset_type


def cosine_similarity(dataset1: sim_dataset_type, dataset2: sim_dataset_type) -> float:
    """
    Calculates the cosine similarity between two datasets. They have to be in the format of ones and zeroes in a list.
    :param dataset1: The first dataset
    :param dataset2: The second dataset
    :return: The cosine similarity
    """
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


def one_hot_encode(keyword_list1, keyword_list2, combined_keyword_list) -> typing.Tuple[typing.List[int], typing.List[int]]:
    """
    Calculates the one-hot-encoded result lists of the two input keyword lists using the combined keyword list as a baseline.
    :param keyword_list1: The first keyword list
    :param keyword_list2: The second keyword list
    :param combined_keyword_list: The combined keyword list
    :return: A tuple with the one-hot-encoded versions of the first and second keyword list
    """
    logger = logging.getLogger(__name__ + '.one_hot_encode')
    logger.debug('calculating list 1 {}, list 2 {} using combined list {}'.format(keyword_list1, keyword_list2, combined_keyword_list))
    result_vector1: typing.List[int] = []
    result_vector2: typing.List[int] = []
    for element in combined_keyword_list:
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


# TODO Refactor. This function could be removed.
def create_keyword_vector(keyword_list1: keyword_dataset_type, keyword_list2: keyword_dataset_type) -> typing.Tuple[
    typing.List[int], typing.List[int]]:
    """
    Creates a tuple of the one-hot-encoded input lists.
    :param keyword_list1: The first input list
    :param keyword_list2: The second input list
    :return: A tuple with the first one-hot-encoded keyword vector and the second one-hot-encoded keyword vector
    """
    logger = logging.getLogger(__name__ + '.create_keyword_vector')
    logger.debug('calculating for {} and {}'.format(keyword_list1, keyword_list2))
    merged_list = list(map(str.lower, (keyword_list1 + keyword_list2)))
    vector = list(set(merged_list))
    logger.debug('calculated combined vector of {}'.format(vector))
    del (merged_list)
    solution = one_hot_encode(keyword_list1, keyword_list2, vector)
    logger.debug('calculated {}'.format(solution))
    return solution


def create_combined_keyword_vector(query: KeywordCoordinate, dataset: dataset_type) -> keyword_dataset_type:
    """
    Creates a combined keyword vector of the query and the dataset. This vector contains all the keywords that appear in either the query or dataset.
    :param query: The query
    :param dataset: The dataset
    :return: A list with all the unique keywords in the query and dataset
    """
    logger = logging.getLogger(__name__ + '.create_combined_keyword_vector')
    logger.debug('calculating for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
    result_keyword_list: keyword_dataset_type = []
    for string in query.keywords:
        result_keyword_list.append(string)
    for kwc in dataset:
        for string in kwc.keywords:
            result_keyword_list.append(string)
    result = list(set(result_keyword_list))
    return result


def separated_cosine_similarity(query_keyword_list: keyword_dataset_type,
                                data_keyword_list: keyword_dataset_type) -> float:
    """
    Calculates the cosine similarity between the keyword list of a query and the keyword list of a data point.
    :param query_keyword_list: The keyword list of the query
    :param data_keyword_list: The keyword list of the data point
    :return: The cosine similarity between the query keywords and data point keywords
    """
    logger = logging.getLogger(__name__ + '.separated_cosine_similarity')
    logger.debug('calculating for query {} and dataset {}'.format(query_keyword_list, data_keyword_list))
    query_vector, data_vector = create_keyword_vector(query_keyword_list, data_keyword_list)
    solution = 1 - cosine_similarity(query_vector, data_vector)
    logger.debug('calculated {}'.format(solution))
    return solution


def combined_cosine_similarity(query_keyword_list: keyword_dataset_type, data_keyword_list: keyword_dataset_type,
                               dataset_keyword_list: keyword_dataset_type) -> float:
    """
    Calculates the cosine similarity between the keyword list of a query and the keyword list of a data point. This happens using a baseline keyword list of the entire dataset.
    :param query_keyword_list: The keyword list of the query
    :param data_keyword_list: The keyword list of the data point
    :param dataset_keyword_list: The keyword list for the entire dataset
    :return: The cosine similarity between the query keywords and data point keywords using the baseline keyword list of the entire dataset.
    """
    logger = logging.getLogger(__name__ + '.combined_cosine_similarity')
    logger.debug('calculating for query {} and dataset {} using combined keyword list {}'.format(query_keyword_list, data_keyword_list, dataset_keyword_list))
    query_vector, data_vector = one_hot_encode(query_keyword_list, data_keyword_list, dataset_keyword_list)
    logger.debug('calculated query vector {} and data vector {}'.format(query_vector, data_vector))
    solution = 1 - cosine_similarity(query_vector, data_vector)
    logger.debug('calculated {}'.format(solution))
    return solution


def word2vec_cosine_similarity(wordlist1: keyword_dataset_type, wordlist2: keyword_dataset_type, model) -> float:
    """
    Calculates the cosine similarity between lists of words based on their word2vec vectors.
    :param wordlist1: The first word list
    :param wordlist2: The second word list
    :param model: The word2vec model
    :return: The calculated keyword similarity cost using word2vec vectors
    """
    logger = logging.getLogger(__name__ + '.word2vec_cosine_similarity')
    logger.debug('calculating for wordlist 1 {} and wordlist 2 {}'.format(wordlist1, wordlist2))
    word_vector_list1: typing.List[np.array] = []
    for element in wordlist1:
        logger.debug('getting vector for word {}'.format(element.lower()))
        try:
            word_vector_list1.append(get_word_vector(element, model))
        except:
            logger.warning('the word {} is not part of the vocabulary and will therefore not be taken into account'.format(element))
    logger.debug('vector 1 {}'.format(word_vector_list1))
    if len(word_vector_list1) == 0:
        logger.error('query (keywords: {}) has no valid keywords'.format(wordlist1))
        raise ValueError('query (keywords: {}) has no valid keywords'.format(wordlist1))
    word_vector_list2: typing.List[np.array] = []
    for element in wordlist2:
        logger.debug('getting vector for word {}'.format(element))
        try:
            word_vector_list2.append(get_word_vector(element, model))
        except:
            logger.warning('the word {} is not part of the vocabulary and will therefore not be taken into account'.format(element))
    logger.debug('vector 2 {}'.format(word_vector_list2))
    if len(word_vector_list2) == 0:
        logger.error('query (keywords: {}) has no valid keywords'.format(wordlist2))
        raise ValueError('query (keywords: {}) has no valid keywords'.format(wordlist2))
    vector_shape = word_vector_list1[0].shape
    logger.debug('vector_shape {}'.format(vector_shape))
    query_vector_sum: np.array = np.zeros(vector_shape)
    for vector in word_vector_list1:
        query_vector_sum = query_vector_sum + vector
    logger.debug('query_vector_sum {}'.format(query_vector_sum))
    subset_vector_sum: np.array = np.zeros(vector_shape)
    for vector in word_vector_list2:
        subset_vector_sum = subset_vector_sum + vector
    logger.debug('subset_vector_sum {}'.format(subset_vector_sum))
    sim = cosine_similarity(query_vector_sum, subset_vector_sum)
    logger.debug('similarity {}'.format(sim))
    logger.debug('returning cost {}'.format(1 - sim))
    return 1 - sim


# https://stackoverflow.com/questions/374626/how-can-i-find-all-the-subsets-of-a-set-with-exactly-n-elements#374645
def find_subsets(input_set: dataset_type, subset_size: int):
    """
    Calculates all the subsets of an input dataset and a given size.
    :param input_set: The input dataset
    :param subset_size: The subset size
    :return: A set of all the subsets
    """
    logger = logging.getLogger(__name__ + '.find_subsets')
    logger.debug('finding all subsets of length {} in set {}'.format(subset_size, dataset_comprehension(input_set)))
    if subset_size > len(input_set):
        solution = set(itertools.combinations(input_set, 0))
    else:
        solution = set(itertools.combinations(input_set, subset_size))
    logger.debug('found {}'.format(sets_of_set_comprehension(solution)))
    return solution

# def find_subsets(input_set: dataset_type, subset_size: int, candidates: pd.DataFrame):
#     """
#     Calculates all the subsets of an input dataset and a given size.
#     :param input_set: The input dataset with candidates to be in solution sets
#     :param subset_size: The subset size
#     :return: A set of all the subsets
#     """
        
#     logger = logging.getLogger(__name__ + '.find_subsets')
#     logger.debug('finding all subsets of length {} in set {}'.format(subset_size, dataset_comprehension(input_set)))
#     if subset_size > len(input_set):
#         solution = set(itertools.combinations(input_set, 0))
#     else:
#         solution = set(itertools.combinations(input_set, subset_size))
#     logger.debug('found {}'.format(sets_of_set_comprehension(solution)))
#     return solution



def get_word_vector(word: str, model):
    """
    Returns the word vector for a given word and model.
    :param word: The word
    :param model: The model
    :return: The vector representation of the word
    """
    word_lower = word.lower()
    return model[word_lower]
