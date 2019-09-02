from costfunctions.costfunction import CostFunction


from metrics.types import distance_function_type, similarity_function_type, dataset_type
from model.keyword_coordinate import KeywordCoordinate


class Type1(CostFunction):
    def __init__(self, distance_metric: distance_function_type, similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float):
        super().__init__(distance_metric, similarity_metric, alpha, beta, omega)

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        return self.alpha * self.get_maximum_for_query(query, dataset) + self.beta * self.get_maximum_for_dataset(dataset) + self.omega * self.get_maximum_keyword_distance(query, dataset)
