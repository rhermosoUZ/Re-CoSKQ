from __future__ import annotations

import logging
import math
import typing

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import normalize_data, denormalize_result_data
from src.metrics.similarity_metrics import find_subsets
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.solver import Solver
from src.utils.logging_utils import dataset_comprehension, result_list_comprehension
from src.utils.types import dataset_type
from src.utils.types import solution_type


class NaiveSolver(Solver):
    """
    The NaiveSolver does not use any kind of heuristic. It calculates the cost for every possibility and returns the best results.
    """
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction, normalize=True, result_length=10):
        """
        Constructs a new NaiveSolver object.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        """
        logger = logging.getLogger(__name__)
        logger.debug('creating with query {}, data {}, cost function {}, normalization {} and result length {}'.format(query, dataset_comprehension(data), cost_function, normalize, result_length))
        super().__init__(query, data, cost_function, normalize, result_length)
        logger.debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> typing.List[solution_type]:
        """
        Implements the solution algorithm.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        logger = logging.getLogger(__name__)
        logger.info('solving for query {} and dataset {} using cost function {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.result_length))
        result_list: typing.List[solution_type] = []
        if(self.normalize_data):
            query, data, self.denormalize_max_x, self.denormalize_min_x, self.denormalize_max_y, self.denormalize_min_y = normalize_data(self.query, self.data)
        else:
            query = self.query
            data = self.data
        for index in range(len(data)):
            list_of_subsets = find_subsets(data, index + 1)
            for subset in list_of_subsets:
                current_cost = self.cost_function.solve(query, subset)
                logger.debug('calculated current cost {}'.format(current_cost))
                if current_cost == math.inf:
                    continue
                if len(result_list) < self.result_length or current_cost < result_list[len(result_list) - 1][0]:
                    result_list.append((current_cost, subset))
                    logger.debug('appended ({}, {}) to result'.format(current_cost, dataset_comprehension(subset)))
                    result_list.sort(key=lambda x: x[0])
                    logger.debug('sorted result {}'.format(result_list_comprehension(result_list)))
                    result_list = result_list[:self.result_length]
        denormalized_result_list = denormalize_result_data(result_list, self.denormalize_max_x, self.denormalize_min_x, self.denormalize_max_y, self.denormalize_min_y)
        logger.info('solved for {} with length {}'.format(result_list_comprehension(denormalized_result_list), self.result_length))
        return denormalized_result_list
