from src.metrics.distance_metrics import euclidean_distance
from src.costfunctions.type3 import Type3
from src.evaluator import Evaluator
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import combined_cosine_similarity, word2vec_cosine_similarity
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_handler import load_word2vec_model, load_pickle
from src.utils.logging_utils import solution_list_comprehension, dataset_comprehension

if __name__ == '__main__':
    # Evaluator, instantiate it first for logging purposes
    ev = Evaluator()

    query = load_pickle('csv_query_data20.pickle')
    print('Query:', query)
    data = load_pickle('data20.pickle')
    print('Data:', dataset_comprehension(data))

    # Define the CostFunctions
    word2vec_model = load_word2vec_model('model_data20.pickle')
    cf1 = Type2(euclidean_distance, combined_cosine_similarity, 0.2, 0.1, 0.7, disable_thresholds=True)
    cf2 = Type3(euclidean_distance, word2vec_cosine_similarity, 0.2, 0.1, 0.7, model=word2vec_model)

    # Choose which Solvers to use
    ns1 = NaiveSolver(query, data, cf1, result_length=5, max_subset_size=4)
    ns2 = NaiveSolver(query, data, cf2, result_length=5, max_subset_size=4)

    # Add Solvers to Evaluator
    ev.add_solver(ns1)
    ev.add_solver(ns2)

    # Run Evaluator and fetch results
    ev.evaluate()
    results = ev.get_results()
    print(solution_list_comprehension(results))
