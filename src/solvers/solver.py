from __future__ import annotations

from costfunctions.costfunction import CostFunction
from metrics.types import dataset_type
from model.keyword_coordinate import KeywordCoordinate
from metrics.types import solution


class Solver:
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction):
        self.query: KeywordCoordinate = query
        self.data: dataset_type = data
        self.cost_function: CostFunction = cost_function

    def solve(self) -> solution:
        pass
