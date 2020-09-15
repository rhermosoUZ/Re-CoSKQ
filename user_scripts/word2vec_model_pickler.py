import os

#import word2vec
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import sys
sys.path.append("..")
from src.utils.data_handler import write_pickle

if __name__ == '__main__':


    # Config
    # Both files should be in the root directory of the project.
    #word2vec_model_name = 'model.bin'
    word2vec_model_name = 'model_test2.bin'
    model_pickle_file_name = 'model.pickle'
    word2vec_model_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../../' + word2vec_model_name)

    # Code - you shouldn't have to make any changes to this
    keyedVectors = KeyedVectors.load(word2vec_model_path, mmap='r')
    print(len(keyedVectors.wv.vocab.items()))
    result_dict = dict()
    for word in keyedVectors.wv.vocab:
        result_dict[word] = keyedVectors.wv.get_vector(word)
    write_pickle(result_dict, model_pickle_file_name)
