from __future__ import annotations

import logging
import typing

from costfunctions.costfunction import CostFunction
from model.keyword_coordinate import KeywordCoordinate
from utils.logging_utils import dataset_comprehension
from utils.types import dataset_type
from utils.types import solution_type


class Solver:
    """
    The Solver solves a given CostFunction for a given query and dataset.
    """
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction, normalize=True, result_length=10):
        """
        Constructs a new Solver object. The Solver class should never be directly instantiated. Instead use a class that inherits from the Solver class and implements the solve() method.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        """
        self.query: KeywordCoordinate = query
        self.data: dataset_type = data
        self.cost_function: CostFunction = cost_function
        self.result_length = result_length
        self.normalize_data = normalize
        self.denormalize_max_x: float = 0.0
        self.denormalize_min_x: float = 0.0
        self.denormalize_max_y: float = 0.0
        self.denormalize_min_y: float = 0.0
        logging.getLogger(__name__).debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> typing.List[solution_type]:
        """
        Implements the solution algorithm. Any solution class needs to implement this.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        pass

    def __str__(self):
        return '{}(query: {}, dataset: {}, cost function: {}, result length {})'.format(type(self).__name__, self.query, dataset_comprehension(self.data), self.cost_function, self.result_length)
