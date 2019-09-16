from __future__ import annotations

import logging
import typing

from metrics.similarity_metrics import create_combined_keyword_vector
from model.keyword_coordinate import KeywordCoordinate
from utils.logging_utils import dataset_comprehension
from utils.types import distance_function_type, similarity_function_type, dataset_type


class CostFunction:
    """
    The CostFunction class acts as base for the specific types of cost functions. It offers all the required methods for the cost calculations. The purpose of a CostFunction is to enable comparability between different sets of data.
    """
    def __init__(self, distance_metric: distance_function_type,
                 similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float, query_distance_threshold: float = 0.7, dataset_distance_threshold: float = 0.7, keyword_similarity_threshold: float = 0.7, disable_thresholds: bool = False) -> typing.NoReturn:
        """
        Constructs a new CostFunction object. The CostFunction class should never be directly instantiated. Instead use a class that inherits from the CostFunction class and implements the solve() method.
        :param distance_metric: The distance metric to calculate coordinate distances between KeywordCoordinates.
        :param similarity_metric: The similarity metric to calculate the similarity between keyword lists of KeywordCoordinates.
        :param alpha: The scaling parameter for the query-dataset distance.
        :param beta: The scaling parameter for the inter-dataset distance.
        :param omega: The scaling parameter for the keyword list similarity.
        :param query_distance_threshold: The threshold for the query-dataset distance.
        :param dataset_distance_threshold: The threshold for the inter-dataset distance.
        :param keyword_similarity_threshold: The threshold for the keyword list similarity.
        :param disable_thresholds: Whether to honor any threshold values.
        """
        self.distance_metric: distance_function_type = distance_metric
        self.similarity_metric: similarity_function_type = similarity_metric
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.query_distance_threshold = query_distance_threshold
        self.dataset_distance_threshold = dataset_distance_threshold
        self.keyword_similarity_threshold = keyword_similarity_threshold
        self.disable_thresholds = disable_thresholds
        logging.getLogger(__name__).debug('created with distance metric {}, similarity metric {}, alpha {}, beta {} and omega {}'.format(self.distance_metric.__name__, self.similarity_metric.__name__, self.alpha, self.beta, self.omega))

    # TODO check if minimum and maximum functions can be refactored into one
    def get_maximum_for_dataset(self, dataset: dataset_type) -> float:
        """
        Calculates the maximum inter-dataset distance cost.
        :param dataset: The dataset.
        :return: Maximum inter-dataset distance cost.
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding maximum distance for dataset {}'.format(dataset_comprehension(dataset)))
        current_maximum: float = 0.0
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates, dataset[index1 + index2 + 1].coordinates)
                if current_value > current_maximum:
                    current_maximum = current_value
        logger.debug('found maximum distance for dataset of {}'.format(current_maximum))
        return current_maximum

    def get_minimum_for_dataset(self, dataset: dataset_type) -> float:
        """
        Calculates the minimum inter-dataset distance cost.
        :param dataset: The dataset.I
        :return: Minimum inter-dataset distance cost.
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding minimum distance for dataset {}'.format(dataset_comprehension(dataset)))
        current_minimum: float = 9999999.9
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates, dataset[index1 + index2 + 1].coordinates)
                if current_value < current_minimum:
                    current_minimum = current_value
        logger.debug('found minimum distance for dataset of {}'.format(current_minimum))
        return current_minimum

    def get_maximum_for_query(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        """
        Calculates the maximum query-dataset distance cost.
        :param query: The query
        :param dataset: The dataset
        :return: Maximum query-dataset distance cost
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding maximum distance for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        current_maximum = 0
        for index in range(len(dataset)):
            current_value = self.distance_metric(query.coordinates, dataset[index].coordinates)
            if current_value > current_maximum:
                current_maximum = current_value
        logger.debug('found maximum distance for query and dataset of {}'.format(current_maximum))
        return current_maximum

    def get_minimum_for_query(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        """
        Calculates the minimum query-dataset distance cost.
        :param query: The query
        :param dataset: The dataset
        :return: Minimum query-dataset distance cost
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding minimum distance for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        current_minimum = 99999999
        for index in range(len(dataset)):
            current_value = self.distance_metric(query.coordinates, dataset[index].coordinates)
            if current_value < current_minimum:
                current_minimum = current_value
        logger.debug('found minimum distance for query and dataset of {}'.format(current_minimum))
        return current_minimum

    def get_maximum_keyword_distance(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        """
        Calculates the maximum keyword distance.
        :param query: The query
        :param dataset: The dataset
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding maximum similarity for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        current_maximum = 0
        combination = False
        if self.similarity_metric.__name__ == 'combined_cosine_similarity':
            combined_keyword_vector: typing.List[str] = create_combined_keyword_vector(query, dataset)
            combination = True
        for element in dataset:
            if combination:
                current_value = self.similarity_metric(query.keywords, element.keywords, combined_keyword_vector)
            else:
                current_value = self.similarity_metric(query.keywords, element.keywords)
            if current_value > current_maximum:
                current_maximum = current_value
        logger.debug('found maximum similarity for query and dataset of {}'.format(current_maximum))
        return current_maximum

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        """
        Implements the solution algorithm. Any costfunction class needs to implement this.
        :param query: The query
        :param dataset: The dataset
        :return: The cost
        """
        pass

    def __str__(self):
        return '{}(dist: {}, sim: {}, alpha: {}, beta: {}, omega: {})'.format(type(self).__name__, self.distance_metric, self.similarity_metric, self.alpha, self.beta, self.omega)
