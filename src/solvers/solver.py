from __future__ import annotations

import logging

from costfunctions.costfunction import CostFunction
from model.keyword_coordinate import KeywordCoordinate
from utils.logging_utils import dataset_comprehension
from utils.types import dataset_type
from utils.types import solution_type


class Solver:
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction):
        self.query: KeywordCoordinate = query
        self.data: dataset_type = data
        self.cost_function: CostFunction = cost_function
        logging.getLogger(__name__).debug('created with query {}, data {} and cost function {}'.format(self.query, dataset_comprehension(self.data), self.cost_function))

    def solve(self) -> solution_type:
        pass

    def __str__(self):
        return '{}(query: {}, dataset: {}, cost function: {})'.format(type(self).__name__, self.query, dataset_comprehension(self.data), self.cost_function)
