from src.costfunctions.type1 import Type1
from src.metrics.distance_metrics import euclidean_distance
from src.metrics.similarity_metrics import word2vec_cosine_similarity
from src.solvers.naive_solver import NaiveSolver
from src.utils.data_handler import load_pickle, write_pickle, load_word2vec_model

if __name__ == '__main__':
    # Config
    file_name_data = 'data20.pickle'
    file_name_query = 'query20.pickle'
    file_name_word2vec_model = 'model_data20.pickle'
    target_file_name = 'precalculated_query_dataset_keyword_similarities_word2vec_data20.pickle'
    max_subset_size = 6
    cost_function = Type1(euclidean_distance, word2vec_cosine_similarity, 0.33, 0.33, 0.33,
                          model=load_word2vec_model(file_name_word2vec_model))
    file_allow_overwrite = False

    # Code
    data = load_pickle(file_name_data)
    query = load_pickle(file_name_query)
    solver = NaiveSolver(query, data, cost_function, max_subset_size=max_subset_size)
    precalculated_query_dataset_distances = solver.get_keyword_similarity()
    write_pickle(precalculated_query_dataset_distances, target_file_name, file_allow_overwrite=file_allow_overwrite)
