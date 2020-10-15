"""
This module generates the word2vec model required by the word2vec cosine similarity.
"""

import os

from gensim.models import word2vec
from gensim.models import Phrases
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
import gensim.downloader as api
from gensim import corpora

if __name__ == '__main__':
    # Config
    # Both files should be in the root directory of the project.
    source_text_file_name = 'text8.txt'
    target_model_file_name = 'model_test2.bin'
    word_vector_size = 100
    allow_to_overwrite_target_model_file = False
    try_to_reuse_word_phrase_file = False
    delete_word_phrase_file_after_generation = True

    # Code - you shouldn't have to make any changes to this
    word_phrase_file_name = 'word_phrases'
    # print (os.path.abspath(os.path.dirname(__file__)) + '/../files/' + source_text_file_name)
    text_file_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../files/' + source_text_file_name)
    word_phrase_file_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../files/' + word_phrase_file_name)
    model_target_file_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../files/' + target_model_file_name)

    if not try_to_reuse_word_phrase_file or not os.path.exists(word_phrase_file_path):
        #word2vec.word2phrase(text_file_path, word_phrase_file_path)
        #sentences = Text8Corpus(datapath(source_text_file_name))
        dataset = api.load("text8")
        dataset = [d for d in dataset]

        dct = corpora.Dictionary(dataset)
        corpus = [dct.doc2bow(line) for line in dataset]
        # Builds bigrams
        bigram = Phrases(dataset, min_count=3, threshold=10)
        #phrases = Phrases(text_file_path, min_count=1, threshold=1)
        print('******OK (Phrases)******')

    if not allow_to_overwrite_target_model_file and os.path.exists(model_target_file_path):
        print('Aborting! Model file exists and config is set to not overwrite it.')
    else:
        dataset = api.load("text8")
        data = [d for d in dataset]
        model = word2vec.Word2Vec(data, size=word_vector_size, min_count = 0)
        model.save(model_target_file_path)  
        #word2vec.word2vec(word_phrase_file_path, model_target_file_path, size=word_vector_size)

    # if delete_word_phrase_file_after_generation:
    #     os.remove(word_phrase_file_path)
