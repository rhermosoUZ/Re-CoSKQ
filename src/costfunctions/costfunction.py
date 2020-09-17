from __future__ import annotations

import logging
import os

import numpy as np

from src.metrics.similarity_metrics import create_combined_keyword_vector
from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.data_handler import load_pickle
from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import distance_function_type, similarity_function_type, dataset_type, \
    precalculated_dict_type, keyword_dataset_type


class CostFunction:
    """
    The CostFunction class acts as base for the specific types of cost functions. It offers all the required methods for the cost calculations. The purpose of a CostFunction is to enable comparability between different sets of data.
    """

    def __init__(self, distance_metric: distance_function_type,
                 similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float,
                 query_distance_threshold: float = 0.7, dataset_distance_threshold: float = 0.7,
                 keyword_similarity_threshold: float = 0.7, disable_thresholds: bool = False, model=None,
                 precalculated_query_dataset_dict: precalculated_dict_type = None,
                 precalculated_inter_dataset_dict: precalculated_dict_type = None,
                 precalculated_keyword_similarity_dict: precalculated_dict_type = None):
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
        :param model: The word2vec model. This can be passed to the CostFunction instead of reading it from disk to improve performance.
        :param precalculated_query_dataset_dict: A dictionary with precalculated query-dataset values for a given frozen subset.
        :param precalculated_inter_dataset_dict: A dictionary with precalculated inter-dataset values for a given frozen subset.
        :param precalculated_keyword_similarity_dict: A dictionary with precalculated keyword similarity values for a given frozen subset.
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
        self.precalculated_query_dataset_dict = precalculated_query_dataset_dict
        self.precalculated_inter_dataset_dict = precalculated_inter_dataset_dict
        self.precalculated_keyword_similarity_dict = precalculated_keyword_similarity_dict
        logger = logging.getLogger(__name__)
        if self.similarity_metric.__name__ == 'word2vec_cosine_similarity':
            try:
                if model is None:
                    model_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../../files/model.pickle')
                    print('*****' + model_path)
                    logger.debug('loading model from path {}'.format(model_path))
                    self.model = load_pickle(model_path)
                else:
                    logger.debug('loading model {} from parameter'.format(model))
                    self.model = model
                    key, value = self.model.popitem()
                    self.model[key] = value
                    # if type(value) != np.ndarray:
                    #     logger.error('Model seems to be corrupt.')
                    #     raise ValueError('Model seems to be corrupt.')
            except:
                logger.error('Could not load model')
                raise ValueError('Could not load model')
        logger.debug('created with distance metric {}, similarity metric {}, alpha {}, beta {} and omega {}'.format(
            self.distance_metric.__name__, self.similarity_metric.__name__, self.alpha, self.beta, self.omega))

    # TODO check if minimum and maximum functions can be refactored into one
    def get_maximum_for_dataset(self, dataset: dataset_type, denormalized_dataset: dataset_type = None) -> float:
        """
        Calculates the maximum inter-dataset distance cost.
        :param dataset: The dataset.
        :param denormalized_dataset: The normalized_dataset. This is used for the matching of precalculated values.
        :return: Maximum inter-dataset distance cost.
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding maximum distance for dataset {}'.format(dataset_comprehension(dataset)))
        if self.precalculated_inter_dataset_dict is not None:
            logger.debug('querying precalculated set')
            if denormalized_dataset is not None:
                dataset_key = denormalized_dataset
            else:
                dataset_key = dataset
            precalculated_result = self.precalculated_inter_dataset_dict.get(frozenset(dataset_key))
            if precalculated_result is not None:
                logger.debug('found precalculated value {}'.format(precalculated_result))
                return precalculated_result
            else:
                logger.warning(
                    'could not find the maximum precalculated inter-dataset value in the given precalculated set. This suggests an erroneous or a wrong dict has been passed into the CostFunction.')
        else:
            logger.debug('No precalculated inter-dataset dict found')
        current_maximum: float = 0.0
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates,
                                                     dataset[index1 + index2 + 1].coordinates)
                if current_value > current_maximum:
                    current_maximum = current_value
        logger.debug('found maximum distance for dataset of {}'.format(current_maximum))
        return current_maximum

    def get_minimum_for_dataset(self, dataset: dataset_type, denormalized_dataset: dataset_type = None) -> float:
        """
        Calculates the minimum inter-dataset distance cost.
        :param dataset: The dataset.
        :param denormalized_dataset: The normalized_dataset. This is used for the matching of precalculated values.
        :return: Minimum inter-dataset distance cost.
        """
        logger = logging.getLogger(__name__)
        logger.debug('finding minimum distance for dataset {}'.format(dataset_comprehension(dataset)))
        if self.precalculated_inter_dataset_dict is not None:
            logger.debug('querying precalculated set')
            if denormalized_dataset is not None:
                dataset_key = denormalized_dataset
            else:
                dataset_key = dataset
            precalculated_result = self.precalculated_inter_dataset_dict.get(frozenset(dataset_key))
            if precalculated_result is not None:
                logger.debug('found precalculated value {}'.format(precalculated_result))
                return precalculated_result
            else:
                logger.warning(
                    'could not find the minimum precalculated inter-dataset value in the given precalculated set. This suggests an erroneous or a wrong dict has been passed into the CostFunction.')
        else:
            logger.debug('No precalculated inter-dataset dict found')
        current_minimum: float = 9999999.9
        if len(dataset) <= 1:
            logger.debug('Dataset of size 1 returning inter-dataset distance of 0.0')
            return 0.0
        for index1 in range(len(dataset)):
            for index2 in range(len(dataset) - index1 - 1):
                current_value = self.distance_metric(dataset[index1].coordinates,
                                                     dataset[index1 + index2 + 1].coordinates)
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
        logger.debug(
            'finding maximum distance for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        if self.precalculated_query_dataset_dict is not None:
            logger.debug('querying precalculated set')
            precalculated_result = self.precalculated_query_dataset_dict.get(frozenset(dataset))
            if precalculated_result is not None:
                logger.debug('found precalculated value {}'.format(precalculated_result))
                return precalculated_result
            else:
                logger.warning(
                    'could not find the maximum precalculated query-dataset value in the given precalculated set. This suggests an erroneous or a wrong dict has been passed into the CostFunction.')
        else:
            logger.debug('No precalculated query-dataset dict found')
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
        logger.debug(
            'finding minimum distance for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        if self.precalculated_query_dataset_dict is not None:
            logger.debug('querying precalculated set')
            precalculated_result = self.precalculated_query_dataset_dict.get(frozenset(dataset))
            if precalculated_result is not None:
                logger.debug('found precalculated value {}'.format(precalculated_result))
                return precalculated_result
            else:
                logger.warning(
                    'could not find the minimum precalculated query-dataset value in the given precalculated set. This suggests an erroneous or a wrong dict has been passed into the CostFunction.')
        else:
            logger.debug('No precalculated query-dataset dict found')
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
        :return: Maximum distance between the keywords
        """
        logger = logging.getLogger(__name__)
        logger.debug(
            'finding maximum similarity for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        if self.precalculated_keyword_similarity_dict is not None:
            logger.debug('querying precalculated set')
            precalculated_result = self.precalculated_keyword_similarity_dict.get(frozenset(dataset))
            if precalculated_result is not None:
                logger.debug('found precalculated value {}'.format(precalculated_result))
                return precalculated_result
            else:
                logger.warning(
                    'could not find the maximum precalculated keyword similarity value in the given precalculated set. This suggests an erroneous or a wrong dict has been passed into the CostFunction.')
        else:
            logger.debug('No precalculated keyword-similarity dict found')
        current_maximum = 0
        combination = False
        latentfactors = False
        if self.similarity_metric.__name__ == 'combined_cosine_similarity':
            combined_keyword_vector: keyword_dataset_type = create_combined_keyword_vector(query, dataset)
            combination = True
        elif self.similarity_metric.__name__ == 'word2vec_cosine_similarity':
            latentfactors = True
        for element in dataset:
            if combination:
                current_value = self.similarity_metric(query.keywords, element.keywords, combined_keyword_vector)
            elif latentfactors:
                current_value = self.similarity_metric(query.keywords, element.keywords, self.model)
            else:
                current_value = self.similarity_metric(query.keywords, element.keywords)
            if current_value > current_maximum:
                current_maximum = current_value
        logger.debug('found maximum similarity cost for query and dataset of {}'.format(current_maximum))
        return current_maximum

    def solve(self, query: KeywordCoordinate, dataset: dataset_type,
              denormalized_dataset: dataset_type = None) -> float:
        """
        Implements the solution algorithm. Any costfunction class needs to implement this.
        :param query: The query
        :param dataset: The dataset
        :param denormalized_dataset: The normalized_dataset. This is used for the matching of precalculated values.
        :return: The cost
        """
        pass

    def __str__(self):
        return '{}(dist: {}, sim: {}, alpha: {}, beta: {}, omega: {})'.format(type(self).__name__, self.distance_metric,
                                                                              self.similarity_metric, self.alpha,
                                                                              self.beta, self.omega)
