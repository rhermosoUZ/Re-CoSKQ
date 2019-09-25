from __future__ import annotations

import logging
import typing

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import normalize_data
from src.metrics.similarity_metrics import find_subsets
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import dataset_type, precalculated_dict_type
from src.utils.typing_definitions import solution_type


class Solver:
    """
    The Solver solves a given CostFunction for a given query and dataset.
    """
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction, normalize=True, result_length=10):
        """
        Constructs a new Solver object. The Solver class should never be directly instantiated. Instead use a class that inherits from the Solver class and implements the solve() method.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        """
        self.query: KeywordCoordinate = query
        self.data: dataset_type = data
        self.cost_function: CostFunction = cost_function
        self.result_length = result_length
        self.normalize_data = normalize
        self.denormalize_max_x: float = 0.0
        self.denormalize_min_x: float = 0.0
        self.denormalize_max_y: float = 0.0
        self.denormalize_min_y: float = 0.0
        logging.getLogger(__name__).debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> typing.List[solution_type]:
        """
        Implements the solution algorithm. Any solution class needs to implement this.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        pass

    def get_max_inter_dataset_distance(self) -> precalculated_dict_type:
        if (self.normalize_data):
            norm_data = normalize_data(self.query, self.data)
            data = norm_data[1]
        else:
            data = self.data
        result_dict: precalculated_dict_type = dict()
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_result = self.cost_function.get_maximum_for_dataset(subset)
                result_dict[frozenset(subset)] = current_result
        return result_dict

    def get_min_inter_dataset_distance(self) -> precalculated_dict_type:
        if (self.normalize_data):
            norm_data = normalize_data(self.query, self.data)
            data = norm_data[1]
        else:
            data = self.data
        result_dict: precalculated_dict_type = dict()
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_result = self.cost_function.get_minimum_for_dataset(subset)
                result_dict[frozenset(subset)] = current_result
        return result_dict

    def get_max_query_dataset_distance(self) -> precalculated_dict_type:
        if (self.normalize_data):
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_result = self.cost_function.get_maximum_for_query(query, data)
                result_dict[frozenset(subset)] = current_result
        return result_dict

    def get_min_query_dataset_distance(self) -> precalculated_dict_type:
        if (self.normalize_data):
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_result = self.cost_function.get_minimum_for_query(query, data)
                result_dict[frozenset(subset)] = current_result
        return result_dict

    def get_max_keyword_similarity(self) -> precalculated_dict_type:
        if (self.normalize_data):
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_result = self.cost_function.get_maximum_keyword_distance(query, data)
                result_dict[frozenset(subset)] = current_result
        return result_dict

    def __str__(self):
        return '{}(query: {}, dataset: {}, cost function: {}, result length {})'.format(type(self).__name__, self.query, dataset_comprehension(self.data), self.cost_function, self.result_length)
