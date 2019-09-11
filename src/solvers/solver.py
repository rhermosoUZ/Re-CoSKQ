from __future__ import annotations

import logging

from costfunctions.costfunction import CostFunction
from model.keyword_coordinate import KeywordCoordinate
from utils.logging_utils import dataset_comprehension
from utils.types import dataset_type
from utils.types import solution_type


class Solver:
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction, normalize=True, result_length=10):
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

    def solve(self) -> solution_type:
        pass

    def __str__(self):
        return '{}(query: {}, dataset: {}, cost function: {}, result length {})'.format(type(self).__name__, self.query, dataset_comprehension(self.data), self.cost_function, self.result_length)
