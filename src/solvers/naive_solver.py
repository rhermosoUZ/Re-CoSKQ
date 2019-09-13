from __future__ import annotations

import logging
import math
import typing

from costfunctions.costfunction import CostFunction
from metrics.distance_metrics import normalize_data, denormalize_result_data
from metrics.similarity_metrics import find_subsets
from model.keyword_coordinate import KeywordCoordinate
from solvers.solver import Solver
from utils.logging_utils import dataset_comprehension, result_list_comprehension
from utils.types import dataset_type
from utils.types import solution_type


class NaiveSolver(Solver):
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction, normalize=True, result_length=10):
        logger = logging.getLogger(__name__)
        logger.debug('creating with query {}, data {}, cost function {}, normalization {} and result length {}'.format(query, dataset_comprehension(data), cost_function, normalize, result_length))
        super().__init__(query, data, cost_function, normalize, result_length)
        logging.getLogger(__name__).debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> typing.List[solution_type]:
        logger = logging.getLogger(__name__)
        logger.debug('solving for query {} and dataset {} using cost function {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.result_length))
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
                if current_cost == math.inf:
                    continue
                if len(result_list) < self.result_length or current_cost < result_list[len(result_list) - 1][0]:
                    result_list.append((current_cost, subset))
                    logger.debug('appended ({}, {}) to result'.format(current_cost, dataset_comprehension(subset)))
                    result_list.sort(key=lambda x: x[0])
                    logger.debug('sorted result {}'.format(result_list_comprehension(result_list)))
                    result_list = result_list[:self.result_length]
        denormalized_result_list = denormalize_result_data(result_list, self.denormalize_max_x, self.denormalize_min_x, self.denormalize_max_y, self.denormalize_min_y)
        logger.debug('solved for {} with length {}'.format(result_list_comprehension(denormalized_result_list), self.result_length))
        return denormalized_result_list
