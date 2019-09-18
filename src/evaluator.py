from __future__ import annotations

import copy
import logging
import logging.config
import os
import typing

from src.solvers.solver import Solver
from src.utils.logging_utils import list_comprehension, solution_list_comprehension
from src.utils.types import solution_type


class Evaluator:
    """
    The Evaluator enables the evaluation of many Solvers at once.
    """
    def __init__(self):
        """
        Constructs a new Evaluator object and initializes the state of the object. Solvers can be added to the evaluator.
        """
        logging.config.fileConfig(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../logs/config/logging.config'))
        self.solvers: typing.List[Solver] = []
        self.results: typing.List[typing.Tuple[solution_type, Solver]] = []
        logging.getLogger(__name__).debug('created')

    def add_solver(self, solver: Solver) -> typing.NoReturn:
        """
        Adds a new solver to the evaluator.
        :param solver: The solver
        """
        self.solvers.append(solver)
        logging.getLogger(__name__).debug('added solver {}'.format(solver))

    def evaluate(self) -> typing.NoReturn:
        """
        Starts the evaluation of all added solvers.
        """
        logger = logging.getLogger(__name__)
        logger.debug('starting evaluation for solvers {}'.format(list_comprehension(self.solvers)))
        for solver in self.solvers:
            result = solver.solve()
            self.results.append((result, solver))
        logger.debug('finished evaluation with results {}'.format(solution_list_comprehension(self.results)))

    def get_results(self) -> typing.List[typing.Tuple[solution_type, Solver]]:
        """
        Retrieves the results of the solutions of all the added solvers.
        :return: A list with tuples. Every tuple contains information about the solution and the used solver.
        """
        logging.getLogger(__name__).debug('getting results {}'.format(solution_list_comprehension(self.results)))
        return copy.deepcopy(self.results)
