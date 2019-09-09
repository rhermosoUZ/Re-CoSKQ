from __future__ import annotations

import logging
import typing

from solvers.solver import Solver
from utils.logging_utils import list_comprehension, solution_list_comprehension
from utils.types import solution_type


class Evaluator:
    def __init__(self):
        self.solvers: typing.List[Solver] = []
        self.results: typing.List[typing.Tuple[solution_type, Solver]] = []
        logging.getLogger(__name__).debug('created')

    def add_solver(self, solver: Solver):
        self.solvers.append(solver)
        logging.getLogger(__name__).debug('added solver {}'.format(solver))

    def evaluate(self):
        logger = logging.getLogger(__name__)
        logger.debug('starting evaluation for solvers {}'.format(list_comprehension(self.solvers)))
        for solver in self.solvers:
            result = solver.solve()
            self.results.append((result, solver))
        logger.debug('finished evaluation with results {}'.format(solution_list_comprehension(self.results)))

    def get_results(self):
        logging.getLogger(__name__).debug('getting results {}'.format(solution_list_comprehension(self.results)))
        return self.results.copy()
