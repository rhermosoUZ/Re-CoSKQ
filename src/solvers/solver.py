from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing as mp

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import normalize_data, denormalize_result_data
from src.metrics.similarity_metrics import find_subsets
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.data_handler import split_subsets
from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import dataset_type, precalculated_dict_type, solution_list


class Solver:
    """
    The Solver solves a given CostFunction for a given query and dataset.
    """

    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction,
                 normalize: bool = True, result_length: int = 10, max_subset_size: int = math.inf):
        """
        Constructs a new Solver object. The Solver class should never be directly instantiated. Instead use a class that inherits from the Solver class and implements the solve() method.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        :param max_subset_size: The maximum size of any subset used to calculate the solution
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
        self.max_subset_size = max_subset_size
        logging.getLogger(__name__).debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> solution_list:
        """
        Implements the solution algorithm. Any solution class needs to implement this.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        pass

    def get_inter_dataset_distance(self) -> precalculated_dict_type:
        """
        Convenience function. Returns the correct inter-dataset distance.
        :return: Correct inter-dataset distance
        """
        return self.get_max_inter_dataset_distance()

    def get_max_inter_dataset_distance(self) -> precalculated_dict_type:
        """
        Calculates a dictionary of the maximum inter-dataset cost for all subsets.
        :return: The Dictionary with frozen subsets as keys and the corresponding cost value as values.
        """
        if self.normalize_data:
            norm_data = normalize_data(self.query, self.data)
            data = norm_data[1]
            denorm_x_max = norm_data[2]
            denorm_x_min = norm_data[3]
            denorm_y_max = norm_data[4]
            denorm_y_min = norm_data[5]
        else:
            data = self.data
        result_dict: precalculated_dict_type = dict()
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        split_ss = split_subsets(list_of_subsets, factor_number_of_processes)
        results = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            for subset in split_ss:
                future = executor.submit(get_max_inter_dataset_distances, self.cost_function, subset)
                results.append(future)
        for result_list in results:
            for subset in result_list.result():
                if self.normalize_data:
                    denormalized_result = denormalize_result_data([(0.0, subset[1])], denorm_x_max, denorm_x_min,
                                                                  denorm_y_max, denorm_y_min)
                    denormalized_subset = denormalized_result[0][1]
                    dict_key = denormalized_subset
                else:
                    dict_key = subset[1]
                result_dict[frozenset(dict_key)] = subset[0]
        return result_dict

    def get_min_inter_dataset_distance(self) -> precalculated_dict_type:
        """
        Calculates a dictionary of the minimum inter-dataset cost for all subsets.
        :return: The Dictionary with frozen subsets as keys and the corresponding cost value as values.
        """
        if self.normalize_data:
            norm_data = normalize_data(self.query, self.data)
            data = norm_data[1]
            denorm_x_max = norm_data[2]
            denorm_x_min = norm_data[3]
            denorm_y_max = norm_data[4]
            denorm_y_min = norm_data[5]
        else:
            data = self.data
        result_dict: precalculated_dict_type = dict()
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        split_ss = split_subsets(list_of_subsets, factor_number_of_processes)
        results = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            for subset in split_ss:
                future = executor.submit(get_min_inter_dataset_distances, self.cost_function, subset)
                results.append(future)
        for result_list in results:
            for subset in result_list.result():
                if self.normalize_data:
                    denormalized_result = denormalize_result_data([(0.0, subset[1])], denorm_x_max, denorm_x_min,
                                                                  denorm_y_max, denorm_y_min)
                    denormalized_subset = denormalized_result[0][1]
                    dict_key = denormalized_subset
                else:
                    dict_key = subset[1]
                result_dict[frozenset(dict_key)] = subset[0]
        return result_dict

    def get_query_dataset_distance(self) -> precalculated_dict_type:
        """
        Convenience function. Returns the correct query-dataset distance.
        :return: Correct query-dataset distance
        """
        if self.cost_function.__class__.__name__ == 'Type3':
            return self.get_min_query_dataset_distance()
        else:
            return self.get_max_query_dataset_distance()

    def get_max_query_dataset_distance(self) -> precalculated_dict_type:
        """
        Calculates a dictionary of the maximum query-dataset cost for all subsets.
        :return: The Dictionary with frozen subsets as keys and the corresponding cost value as values.
        """
        if self.normalize_data:
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        split_ss = split_subsets(list_of_subsets, factor_number_of_processes)
        results = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            for subset in split_ss:
                future = executor.submit(get_max_query_dataset_distances, self.cost_function, query, subset)
                results.append(future)
        for result_list in results:
            for subset in result_list.result():
                result_dict[frozenset(subset[1])] = subset[0]
        return result_dict

    def get_min_query_dataset_distance(self) -> precalculated_dict_type:
        """
        Calculates a dictionary of the minimum query-dataset cost for all subsets.
        :return: The Dictionary with frozen subsets as keys and the corresponding cost value as values.
        """
        if self.normalize_data:
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        split_ss = split_subsets(list_of_subsets, factor_number_of_processes)
        results = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            for subset in split_ss:
                future = executor.submit(get_min_query_dataset_distances, self.cost_function, query, subset)
                results.append(future)
        for result_list in results:
            for subset in result_list.result():
                result_dict[frozenset(subset[1])] = subset[0]
        return result_dict

    def get_keyword_similarity(self) -> precalculated_dict_type:
        """
        Convenience function. Returns the correct keyword similarity.
        :return: Correct keyword similarity
        """
        return self.get_max_keyword_similarity()

    def get_max_keyword_similarity(self) -> precalculated_dict_type:
        """
        Calculates a dictionary of the maximum keyword-similarity cost for all subsets.
        :return: The Dictionary with frozen subsets as keys and the corresponding cost value as values.
        """
        if self.normalize_data:
            norm_data = normalize_data(self.query, self.data)
            query = norm_data[0]
            data = norm_data[1]
        else:
            query = self.query
            data = self.data
        result_dict: precalculated_dict_type = dict()
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        split_ss = split_subsets(list_of_subsets, factor_number_of_processes)
        results = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            for subset in split_ss:
                future = executor.submit(get_max_keyword_similarity, self.cost_function, query, subset)
                results.append(future)
        for result_list in results:
            for subset in result_list.result():
                result_dict[frozenset(subset[1])] = subset[0]
        return result_dict

    def get_all_subsets(self, data):
        """
        Calculates all the possible subsets for the given data. Takes the set maximum length for subsets into account.
        :param data: The data
        :return: A list of all possible subsets
        """
        max_length = min(len(data), self.max_subset_size)
        list_of_subsets = []
        for index in range(max_length):
            new_subsets = find_subsets(data, index + 1)
            for subset in new_subsets:
                list_of_subsets.append(subset)
        return list_of_subsets

    def __str__(self):
        return '{}(query: {}, dataset: {}, cost function: {}, result length {})'.format(type(self).__name__, self.query, dataset_comprehension(self.data), self.cost_function, self.result_length)


def get_max_inter_dataset_distances(costfunction: CostFunction, subsets):
    """
    This function gets executed inside every maximum inter-dataset distance process.
    :param costfunction: The CostFunction
    :param subsets: The subsets for the process
    :return: A list with tuples of the costs and their corresponding subset
    """
    results = []
    for subset in subsets:
        current_cost = costfunction.get_maximum_for_dataset(subset)
        results.append((current_cost, subset))
    return results


def get_min_inter_dataset_distances(costfunction: CostFunction, subsets):
    """
    This function gets executed inside every minimum inter-dataset distance process.
    :param costfunction: The CostFunction
    :param subsets: The subsets for the process
    :return: A list with tuples of the costs and their corresponding subset
    """
    results = []
    for subset in subsets:
        current_cost = costfunction.get_minimum_for_dataset(subset)
        results.append((current_cost, subset))
    return results


def get_max_query_dataset_distances(costfunction: CostFunction, query: KeywordCoordinate, subsets):
    """
    This function gets executed inside every maximum query-dataset distance process.
    :param costfunction: The CostFunction
    :param query: The Query
    :param subsets: The subsets for the process
    :return: A list with tuples of the costs and their corresponding subset
    """
    results = []
    for subset in subsets:
        current_cost = costfunction.get_maximum_for_query(query, subset)
        results.append((current_cost, subset))
    return results


def get_min_query_dataset_distances(costfunction: CostFunction, query: KeywordCoordinate, subsets):
    """
    This function gets executed inside every minimum query-dataset distance process.
    :param costfunction: The CostFunction
    :param query: The Query
    :param subsets: The subsets for the process
    :return: A list with tuples of the costs and their corresponding subset
    """
    results = []
    for subset in subsets:
        current_cost = costfunction.get_minimum_for_query(query, subset)
        results.append((current_cost, subset))
    return results


def get_max_keyword_similarity(costfunction: CostFunction, query: KeywordCoordinate, subsets):
    """
    This function gets executed inside every maximum keyword similarity process.
    :param costfunction: The CostFunction
    :param query: The Query
    :param subsets: The subsets for the process
    :return: A list with tuples of the costs and their corresponding subset
    """
    results = []
    for subset in subsets:
        current_cost = costfunction.get_maximum_keyword_distance(query, subset)
        results.append((current_cost, subset))
    return results
