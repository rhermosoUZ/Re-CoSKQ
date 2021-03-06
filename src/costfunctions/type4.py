import logging
import math

from src.costfunctions.costfunction import CostFunction
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import distance_function_type, similarity_function_type, dataset_type, \
    precalculated_dict_type


class Type4(CostFunction):
    # TODO check if this works as expected.
    def __init__(self, distance_metric: distance_function_type, similarity_metric: similarity_function_type,
                 alpha: float, beta: float, omega: float, phi_1: float, phi_2: float,
                 query_distance_threshold: float = 0.7, dataset_distance_threshold: float = 0.7,
                 keyword_similarity_threshold: float = 0.7, disable_thresholds: bool = False, model=None,
                 precalculated_query_dataset_dict: precalculated_dict_type = None,
                 precalculated_inter_dataset_dict: precalculated_dict_type = None,
                 precalculated_keyword_similarity_dict: precalculated_dict_type = None):
        """
        Constructs a Type2 cost function object.
        :param distance_metric: The distance metric to calculate coordinate distances between KeywordCoordinates.
        :param similarity_metric: The similarity metric to calculate the similarity between keyword lists of KeywordCoordinates.
        :param alpha: The scaling parameter for the query-dataset distance.
        :param beta: The scaling parameter for the inter-dataset distance.
        :param omega: The scaling parameter for the keyword list similarity.
        :param phi_1: The first tuning parameter of the unified cost function.
        :param phi_2: The second tuning parameter of the unified cost function.
        :param query_distance_threshold: The threshold for the query-dataset distance.
        :param dataset_distance_threshold: The threshold for the inter-dataset distance.
        :param keyword_similarity_threshold: The threshold for the keyword list similarity.
        :param disable_thresholds: Whether to honor any threshold values.
        :param model: The word2vec model. This can be passed to the CostFunction instead of reading it from disk to improve performance.
        :param precalculated_query_dataset_dict: A dictionary with precalculated query-dataset values for a given frozen subset.
        :param precalculated_inter_dataset_dict: A dictionary with precalculated inter-dataset values for a given frozen subset.
        :param precalculated_keyword_similarity_dict: A dictionary with precalculated keyword similarity values for a given frozen subset.
        """
        super().__init__(distance_metric, similarity_metric, alpha, beta, omega, query_distance_threshold,
                         dataset_distance_threshold, keyword_similarity_threshold, disable_thresholds, model,
                         precalculated_query_dataset_dict, precalculated_inter_dataset_dict,
                         precalculated_keyword_similarity_dict)
        self.phi_1 = phi_1
        self.phi_2 = phi_2

    def solve(self, query: KeywordCoordinate, dataset: dataset_type,
              denormalized_dataset: dataset_type = None) -> float:
        """
        Solves the Type4 cost function.
        :param query: The query
        :param dataset: The dataset
        :param denormalized_dataset: The normalized_dataset. This is used for the matching of precalculated values.
        :return: The maximum cost for the given query and dataset
        """
        logger = logging.getLogger(__name__)
        logger.debug('solving for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        # TODO does this type of threshold filtering make sense for the unified function?
        query_distance = self.get_maximum_for_query(query, dataset)
        logger.debug('solved query distance for {}'.format(query_distance))
        if denormalized_dataset is not None:
            dataset_distance = self.get_maximum_for_dataset(dataset, denormalized_dataset)
        else:
            dataset_distance = self.get_maximum_for_dataset(dataset)
        logger.debug('solved dataset distance for {}'.format(dataset_distance))
        keyword_similarity = self.get_maximum_keyword_distance(query, dataset)
        logger.debug('solved keyword similarity for {}'.format(keyword_similarity))
        if (not self.disable_thresholds and (
                query_distance > self.query_distance_threshold or dataset_distance > self.dataset_distance_threshold or keyword_similarity > self.keyword_similarity_threshold)):
            logger.debug(
                'One of the thresholds was not met. Query threshold: {}, dataset threshold: {}, keyword threshold {}'.format(
                    self.query_distance_threshold, self.dataset_distance_threshold, self.keyword_similarity_threshold))
            return math.inf
        else:
            a: float = 0.0
            for element in dataset:
                a += self.distance_metric(query.coordinates, element.coordinates) ** self.phi_1
            a = a ** (1 / self.phi_1)
            a = (self.alpha * a) ** self.phi_2
            b: float = (self.beta * dataset_distance) ** self.phi_2
            c: float = ((self.omega * keyword_similarity) ** self.phi_2) ** (
                        1 / self.phi_2)
            solution = a + b + c
            logger.debug('solved with a cost of {}'.format(solution))
            return solution

    def __str__(self):
        return 'Type4(dist: {}, sim: {}, alpha: {}, beta: {}, omega: {}, phi_1: {}, phi_2: {})'.format(self.distance_metric, self.similarity_metric, self.alpha, self.beta, self.omega, self.phi_1, self.phi_2)
