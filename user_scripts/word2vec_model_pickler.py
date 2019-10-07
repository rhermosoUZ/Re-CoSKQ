import os

import word2vec

from src.utils.data_handler import write_pickle

if __name__ == '__main__':
    # Config
    # Both files should be in the root directory of the project.
    word2vec_model_name = 'model.bin'
    model_pickle_file_name = 'model.pickle'
    word2vec_model_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../../' + word2vec_model_name)

    # Code - you shouldn't have to make any changes to this
    model = word2vec.load(word2vec_model_path)
    result_dict = dict()
    for word in model.vocab:
        result_dict[word] = model.get_vector(word)
    write_pickle(result_dict, model_pickle_file_name)
