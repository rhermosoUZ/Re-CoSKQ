from __future__ import annotations

import typing

from model.coordinate import Coordinate
from model.keyword_coordinate import KeywordCoordinate

# from solvers.solver import Solver


distance_function_type = typing.Callable[[Coordinate, Coordinate], float]
similarity_function_type = typing.Callable[[typing.List[str], typing.List[str]], float]

dataset_type = typing.List[KeywordCoordinate]
sim_dataset_type = typing.List[int]
sim_tuple_type = typing.Tuple[sim_dataset_type, sim_dataset_type]
keyword_dataset_type = typing.List[str]
solution_type = typing.Tuple[float, typing.List[KeywordCoordinate]]
# solution_list = typing.List[typing.Tuple[solution_type, Solver]]

