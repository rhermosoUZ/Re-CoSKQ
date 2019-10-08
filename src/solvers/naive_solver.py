from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing as mp

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import normalize_data, denormalize_result_data
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.solver import Solver
from src.utils.data_handler import split_subsets
from src.utils.logging_utils import dataset_comprehension, result_list_comprehension
from src.utils.typing_definitions import dataset_type, solution_list


class NaiveSolver(Solver):
    """
    The NaiveSolver does not use any kind of heuristic. It calculates the cost for every possibility and returns the best results.
    """

    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction,
                 normalize: bool = True, result_length: int = 10, max_subset_size: int = math.inf):
        """
        Constructs a new NaiveSolver object.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        :param max_subset_size: The maximum size of any subset used to calculate the solution
        """
        logger = logging.getLogger(__name__)
        logger.debug('creating with query {}, data {}, cost function {}, normalization {} and result length {}'.format(query, dataset_comprehension(data), cost_function, normalize, result_length))
        super().__init__(query, data, cost_function, normalize, result_length, max_subset_size)
        logger.debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> solution_list:
        """
        Implements the solution algorithm.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        logger = logging.getLogger(__name__)
        logger.info('solving for query {} and dataset {} using cost function {} and result length {}'.format(self.query,
                                                                                                             dataset_comprehension(
                                                                                                                 self.data),
                                                                                                             self.cost_function,
                                                                                                             self.result_length))
        result_list: solution_list = []
        if (self.normalize_data):
            query, data, self.denormalize_max_x, self.denormalize_min_x, self.denormalize_max_y, self.denormalize_min_y = normalize_data(
                self.query, self.data)
        else:
            query = self.query
            data = self.data
        list_of_subsets = self.get_all_subsets(data)
        factor_number_of_processes: int = 2
        list_of_split_subsets = split_subsets(list_of_subsets,
                                              scaling_factor_number_of_processes=factor_number_of_processes)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp.cpu_count() * factor_number_of_processes) as executor:
            future_list = []
            for subsets in list_of_split_subsets:
                future = executor.submit(get_cost_for_subset, query, subsets, self.cost_function)
                future_list.append(future)
            for future in future_list:
                for solution in future.result():
                    result_list.append(solution)
        result_list.sort(key=lambda x: x[0])
        result_list = result_list[:self.result_length]
        denormalized_result_list = denormalize_result_data(result_list, self.denormalize_max_x, self.denormalize_min_x,
                                                           self.denormalize_max_y, self.denormalize_min_y)
        logger.info('solved for {} with length {}'.format(result_list_comprehension(denormalized_result_list),
                                                          self.result_length))
        return denormalized_result_list


def get_cost_for_subset(query, subsets, costfunction) -> solution_list:
    """
    Calculates the costs of all the subsets for a given query and cost function.
    :param query: The query
    :param subsets: The list of subsets
    :param costfunction: The costfunction
    :return: A list of solutions. Each solution being a cost and the corresponding subset
    """
    results: solution_list = []
    for subset in subsets:
        current_result = costfunction.solve(query, subset)
        results.append((current_result, subset))
    return results
