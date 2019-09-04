from __future__ import annotations

from metrics.types import solution
from metrics.types import dataset_type
from costfunctions.costfunction import CostFunction
from model.keyword_coordinate import KeywordCoordinate
from metrics.similarity_metrics import find_subsets
from solvers.solver import Solver


class NaiveSolver(Solver):
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction):
        super().__init__(query, data, cost_function)

    def solve(self) -> solution:
        lowest_cost = 999999999
        lowest_cost_set = {None, None}
        for index in range(len(self.data)):
            list_of_subsets = find_subsets(self.data, index + 1)
            for subset in list_of_subsets:
                current_cost = self.cost_function.solve(self.query, subset)
                if current_cost < lowest_cost:
                    lowest_cost = current_cost
                    lowest_cost_set = subset
        return (lowest_cost, lowest_cost_set)

    def __str__(self):
        return 'NaiveSolver'
