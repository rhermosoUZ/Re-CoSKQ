import sys
import time

sys.path.append("..")

from src.utils.data_handler import write_pickle, load_word2vec_model, calculate_model_subset, load_pickle

if __name__ == '__main__':
    
    start_time = time.time()
    
    # Config
    # Both files should be in the root directory of the project.
    word2vec_model_name = 'model.pickle'
    model_pickle_file_name = 'data20_model.pickle'
    query_file_name = 'data20_query.pickle'
    data_file_name = 'data20_dataset.pickle'
    file_allow_overwrite = True

    # Code - you shouldn't have to make any changes to this
    model = load_word2vec_model(word2vec_model_name)
    query = load_pickle(query_file_name)
    data = load_pickle(data_file_name)
    shrunk_model = calculate_model_subset(query, data, model)
    write_pickle(shrunk_model, model_pickle_file_name, file_allow_overwrite=file_allow_overwrite)

    print("--- %s seconds ---" % (time.time() - start_time))