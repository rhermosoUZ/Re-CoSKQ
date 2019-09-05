from costfunctions.costfunction import CostFunction


from metrics.types import distance_function_type, similarity_function_type, dataset_type
from model.keyword_coordinate import KeywordCoordinate


class Type4(CostFunction):
    def __init__(self, distance_metric: distance_function_type, similarity_metric: similarity_function_type, alpha: float, beta: float, omega: float, phi_1: float, phi_2: float):
        super().__init__(distance_metric, similarity_metric, alpha, beta, omega)
        self.phi_1 = phi_1
        self.phi_2 = phi_2

    def solve(self, query: KeywordCoordinate, dataset: dataset_type) -> float:
        a: float = 0.0
        for element in dataset:
            a += self.distance_metric(query.coordinates, element.coordinates) ** self.phi_1
        a = a ** (1 / self.phi_1)
        a = (self.alpha * a) ** self.phi_2
        b: float = (self.beta * self.get_maximum_for_dataset(dataset)) ** self.phi_2
        c: float = ((self.omega * self.get_maximum_keyword_distance(query, dataset)) ** self.phi_2) ** (1 / self.phi_2)
        return a + b + c
