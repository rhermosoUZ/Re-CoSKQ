from __future__ import annotations

import typing

from metrics.types import distance_function_type, similarity_function_type, dataset_type
from model.keyword_coordinate import KeywordCoordinate


class CostFunction:
    def __init__(self, distance_metric: distance_function_type,
                 similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float) -> typing.NoReturn:
        self.distance_metric: distance_function_type = distance_metric
        self.similarity_metric: similarity_function_type = similarity_metric
        self.alpha = alpha
        self.beta = beta
        self.omega = omega

    # TODO check if both dataset functions can be refactored into one
    def get_maximum_for_dataset(self, dataset: dataset_type) -> float:
        current_maximum: float = 0.0
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates, dataset[index1 + index2 + 1].coordinates)
                if current_value > current_maximum:
                    current_maximum = current_value
        return current_maximum

    def get_minimum_for_dataset(self, dataset: dataset_type) -> float:
        current_minimum: float = 9999999.9
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates, dataset[index1 + index2 + 1].coordinates)
                if current_value < current_minimum:
                    current_minimum = current_value
        return current_minimum

    def get_maximum_for_query(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        current_maximum = 0
        for index in range(len(dataset)):
            current_value = self.distance_metric(query.coordinates, dataset[index].coordinates)
            if current_value > current_maximum:
                current_maximum = current_value
        return current_maximum

    def get_minimum_for_query(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        current_minimum = 99999999
        for index in range(len(dataset)):
            current_value = self.distance_metric(query.coordinates, dataset[index].coordinates)
            if current_value < current_minimum:
                current_minimum = current_value
        return current_minimum

    def get_maximum_keyword_distance(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        current_maximum = 0
        for element in dataset:
            current_value = self.similarity_metric(query.keywords, element.keywords)
            if current_value > current_maximum:
                current_maximum = current_value
        return current_maximum

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        pass
