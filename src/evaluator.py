from __future__ import annotations

from solvers.solver import Solver
import typing
from metrics.types import solution


class Evaluator:
    def __init__(self):
        self.solvers: typing.List[Solver] = []
        self.results: typing.List[typing.Tuple[solution, Solver]] = []

    def add_solver(self, solver: Solver):
        self.solvers.append(solver)

    def evaluate(self):
        for solver in self.solvers:
            result = solver.solve()
            self.results.append((result, solver))

    def get_results(self):
        return self.results.copy()