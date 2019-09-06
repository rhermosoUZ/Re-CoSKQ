from __future__ import annotations

from solvers.solver import Solver
import typing
import logging
from metrics.types import solution_type


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
        logger.debug('starting evaluation for solvers {}'.format(self.solvers))
        for solver in self.solvers:
            result = solver.solve()
            self.results.append((result, solver))
        logger.debug('finished evaluation with results {}'.format(self.results))

    def get_results(self):
        logging.getLogger(__name__).debug('getting results {}'.format(self.results))
        return self.results.copy()
