import logging
import math

from src.costfunctions.costfunction import CostFunction
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension
from src.utils.types import distance_function_type, similarity_function_type, dataset_type


class Type1(CostFunction):
    def __init__(self, distance_metric: distance_function_type, similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float, query_distance_threshold: float = 0.7, dataset_distance_threshold: float = 0.7, keyword_similarity_threshold: float = 0.7, disable_thresholds: bool = False):
        """
        Constructs a Type1 cost function object.
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
        super().__init__(distance_metric, similarity_metric, alpha, beta, omega, query_distance_threshold, dataset_distance_threshold, keyword_similarity_threshold, disable_thresholds)

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        """
        Solves the Type1 cost function.
        :param query: The query
        :param dataset: The dataset
        :return: The maximum cost for the given query and dataset
        """
        logger = logging.getLogger(__name__)
        logger.info('solving for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        query_distance = self.get_maximum_for_query(query, dataset)
        logger.debug('solved query distance for {}'.format(query_distance))
        dataset_distance = self.get_maximum_for_dataset(dataset)
        logger.debug('solved dataset distance for {}'.format(dataset_distance))
        keyword_similarity = self.get_maximum_keyword_distance(query, dataset)
        logger.debug('solved keyword similarity for {}'.format(keyword_similarity))
        if (not self.disable_thresholds and (query_distance > self.query_distance_threshold or dataset_distance > self.dataset_distance_threshold or keyword_similarity > self.keyword_similarity_threshold)):
            logger.info('One of the thresholds was not met. Query threshold: {}, dataset threshold: {}, keyword threshold {}'.format(self.query_distance_threshold, self.dataset_distance_threshold, self.keyword_similarity_threshold))
            return math.inf
        else:
            solution = self.alpha * query_distance + self.beta * dataset_distance + self.omega * keyword_similarity
            logger.info('solved with a cost of {}'.format(solution))
            return solution
