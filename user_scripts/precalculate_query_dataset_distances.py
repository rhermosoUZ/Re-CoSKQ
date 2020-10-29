import time

from src.costfunctions.type1 import Type1
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import combined_cosine_similarity
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_handler import load_pickle, write_pickle

if __name__ == '__main__':
    start_time = time.time()
    # Config
    file_name_data = 'data20_dataset.pickle'
    file_name_query = 'data20_query.pickle'
    target_file_name = 'precalculated_query_dataset_distances_data20.pickle'
    max_subset_size = 3
    cost_function = Type1(euclidean_distance, combined_cosine_similarity, 0.33, 0.33, 0.33)
    file_allow_overwrite = True

    # Code
    data = load_pickle(file_name_data)
    
    query = load_pickle(file_name_query)
    solver = NaiveSolver(query, data, cost_function, max_subset_size=max_subset_size)
    precalculated_query_dataset_distances = solver.get_query_dataset_distance()
    write_pickle(precalculated_query_dataset_distances, target_file_name, file_allow_overwrite=file_allow_overwrite)

    print("--- %s seconds ---" % (time.time() - start_time))