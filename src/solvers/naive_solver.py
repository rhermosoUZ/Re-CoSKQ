from __future__ import annotations

from metrics.types import solution_type
from metrics.types import dataset_type
from costfunctions.costfunction import CostFunction
from model.keyword_coordinate import KeywordCoordinate
from metrics.similarity_metrics import find_subsets
from solvers.solver import Solver
import logging


class NaiveSolver(Solver):
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction):
        logger = logging.getLogger(__name__)
        logger.debug('creating with query {}, data {} and cost function {}'.format(query, data, cost_function))
        super().__init__(query, data, cost_function)
        logging.getLogger(__name__).debug('created with query {}, data {} and cost function {}'.format(self.query, self.data, self.cost_function))

    def solve(self) -> solution_type:
        logger = logging.getLogger(__name__)
        logger.debug('solving for query {} and dataset {} using cost function {}'.format(self.query, self.data, self.cost_function))
        lowest_cost = 999999999
        lowest_cost_set = {None, None}
        for index in range(len(self.data)):
            list_of_subsets = find_subsets(self.data, index + 1)
            for subset in list_of_subsets:
                current_cost = self.cost_function.solve(self.query, subset)
                if current_cost < lowest_cost:
                    lowest_cost = current_cost
                    lowest_cost_set = subset
        solution = (lowest_cost, lowest_cost_set)
        logger.debug('solved for {}'.format(solution))
        return solution

    def __str__(self):
        return 'NaiveSolver'
