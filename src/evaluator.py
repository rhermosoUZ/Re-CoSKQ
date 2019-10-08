from __future__ import annotations

import concurrent.futures
import copy
import logging
import logging.config
import os
import time
import typing

from src.solvers.solver import Solver
from src.utils.logging_utils import list_comprehension, solution_list_comprehension
from src.utils.typing_definitions import solution_type


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
        self.timings: typing.List[typing.Tuple[float, Solver]] = []
        logging.getLogger(__name__).debug('created')

    def add_solver(self, solver: Solver) -> typing.NoReturn:
        """
        Adds a new solver to the evaluator.
        :param solver: The solver
        """
        self.solvers.append(solver)
        logging.getLogger(__name__).info('added solver {}'.format(solver))

    def reset(self):
        """
        Resets the state of the Evaluator. This clears all the solvers, results and timings.
        """
        self.solvers: typing.List[Solver] = []
        self.results: typing.List[typing.Tuple[solution_type, Solver]] = []
        self.timings: typing.List[float] = []

    def evaluate(self, evaluate_all_solvers_concurrently=False) -> typing.NoReturn:
        """
        Starts the evaluation of all added solvers.
        :param evaluate_all_solvers_concurrently: Flag for evaluation if all passed solvers should be evaluated concurrently. WARNING: This potentially requires a lot of memory.
        """
        logger = logging.getLogger(__name__)
        logger.info('starting evaluation for solvers {}'.format(list_comprehension(self.solvers)))
        self.results: typing.List[typing.Tuple[solution_type, Solver]] = []
        self.timings: typing.List[float] = []
        if evaluate_all_solvers_concurrently:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_list = []
                for solver in self.solvers:
                    future = executor.submit(solver.solve)
                    future_list.append(future)
                for index in range(len(future_list)):
                    self.results.append((future_list[index].result(), self.solvers[index]))
        else:
            for solver in self.solvers:
                t_started = time.time()
                current_result = solver.solve()
                t_finished = time.time()
                self.results.append((current_result, solver))
                self.timings.append(((t_finished - t_started), solver))
        logger.info('finished evaluation with results {}'.format(solution_list_comprehension(self.results)))

    def get_results(self) -> typing.List[typing.Tuple[solution_type, Solver]]:
        """
        Retrieves the results of the solutions of all the added solvers.
        :return: A list with tuples. Every tuple contains information about the solution and the used solver.
        """
        logging.getLogger(__name__).debug('getting results {}'.format(solution_list_comprehension(self.results)))
        return copy.deepcopy(self.results)

    def get_timings(self):
        """
        Retrieves the timings of the solutions of all the added solvers.
        :return: A list with tuples. Every tuple contains the timing and the used solver.
        """
        logging.getLogger(__name__).debug('getting timings {}'.format(self.timings))
        return copy.deepcopy(self.timings)
