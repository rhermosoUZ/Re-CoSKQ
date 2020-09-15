import multiprocessing as mp

from src.costfunctions.type1 import Type1
from src.costfunctions.type2 import Type2
from src.evaluator import Evaluator
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import combined_cosine_similarity, word2vec_cosine_similarity
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_handler import load_word2vec_model, load_pickle
from src.utils.logging_utils import solution_list_comprehension, dataset_comprehension, timing_list_comprehension
from src.utils.typing_definitions import dataset_type

if __name__ == '__main__':
    # Evaluator, instantiate it first for logging purposes
    ev = Evaluator()

    query: KeywordCoordinate = load_pickle('synthetic_query.pickle')
    print('Query:', query)
    data: dataset_type = load_pickle('synthetic20_dataset.pickle')
    print('Data:', dataset_comprehension(data))

    # Load precalculated values and models
    precalculated_inter_dataset_distances_data20 = load_pickle('precalculated_inter_dataset_distances_data20.pickle')
    precalculated_query_dataset_distances_data20 = load_pickle('precalculated_query_dataset_distances_data20.pickle')
    precalculated_query_dataset_keyword_similarities_data20 = load_pickle(
        'precalculated_query_dataset_keyword_similarities_data20.pickle')
    
    # **** ONLY FOR word2vec model executions
    # precalculated_query_dataset_keyword_similarities_word2vec_data20 = load_pickle(
    #    'precalculated_query_dataset_keyword_similarities_word2vec_data20.pickle')
    # word2vec_model = load_word2vec_model('data20_model.pickle')
    # ****

    # Define the CostFunctions. For all possible parameters refer to the documentation.
    cf1 = Type1(euclidean_distance, combined_cosine_similarity, 0.2, 0.1, 0.7)
#    cf2 = Type2(euclidean_distance, word2vec_cosine_similarity, 0.2, 0.1, 0.7, model=word2vec_model)
    cf3 = Type1(euclidean_distance, combined_cosine_similarity, 0.2, 0.1, 0.7,
                precalculated_inter_dataset_dict=precalculated_inter_dataset_distances_data20,
                precalculated_query_dataset_dict=precalculated_query_dataset_distances_data20,
                precalculated_keyword_similarity_dict=precalculated_query_dataset_keyword_similarities_data20
                )
#    cf4 = Type2(euclidean_distance, word2vec_cosine_similarity, 0.2, 0.1, 0.7,
#                precalculated_inter_dataset_dict=precalculated_inter_dataset_distances_data20,
#                precalculated_keyword_similarity_dict=precalculated_query_dataset_keyword_similarities_word2vec_data20,
#                model=word2vec_model)

    # Choose which Solvers to use. For all possible parameters refer to the documentation.
    max_number_of_processes = mp.cpu_count()
    ns1 = NaiveSolver(query, data, cf1, result_length=5, max_subset_size=6,
                      max_number_of_concurrent_processes=max_number_of_processes)
#    ns2 = NaiveSolver(query, data, cf2, result_length=5, max_subset_size=6,
#                      max_number_of_concurrent_processes=max_number_of_processes)
    ns3 = NaiveSolver(query, data, cf3, result_length=5, max_subset_size=6,
                      max_number_of_concurrent_processes=max_number_of_processes)
#    ns4 = NaiveSolver(query, data, cf4, result_length=5, max_subset_size=6,
#                      max_number_of_concurrent_processes=max_number_of_processes)

    # Add Solvers to Evaluator
    ev.add_solver(ns1)
#    ev.add_solver(ns2)
    ev.add_solver(ns3)
#    ev.add_solver(ns4)

    # Run Evaluator and fetch results
    ev.evaluate()
    results = ev.get_results()
    timings = ev.get_timings()
    print(solution_list_comprehension(results))
    print(timing_list_comprehension(timings))
