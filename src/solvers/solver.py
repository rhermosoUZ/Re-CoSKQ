from __future__ import annotations

from costfunctions.costfunction import CostFunction
from utils.types import dataset_type
from model.keyword_coordinate import KeywordCoordinate
from utils.types import solution_type
from utils.logging_utils import dataset_comprehension
import logging


class Solver:
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction):
        self.query: KeywordCoordinate = query
        self.data: dataset_type = data
        self.cost_function: CostFunction = cost_function
        logging.getLogger(__name__).debug('created with query {}, data {} and cost function {}'.format(self.query, dataset_comprehension(self.data), self.cost_function))

    def solve(self) -> solution_type:
        pass
