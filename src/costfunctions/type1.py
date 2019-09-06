from costfunctions.costfunction import CostFunction


from utils.types import distance_function_type, similarity_function_type, dataset_type
from utils.logging_utils import dataset_comprehension
from model.keyword_coordinate import KeywordCoordinate
import logging


class Type1(CostFunction):
    def __init__(self, distance_metric: distance_function_type, similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float):
        super().__init__(distance_metric, similarity_metric, alpha, beta, omega)

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        logger = logging.getLogger(__name__)
        logger.debug('solving for query {} and dataset {}'.format(query, dataset_comprehension(dataset)))
        solution = self.alpha * self.get_maximum_for_query(query, dataset) + self.beta * self.get_maximum_for_dataset(dataset) + self.omega * self.get_maximum_keyword_distance(query, dataset)
        logger.debug('solved with a cost of {}'.format(solution))
        return solution
