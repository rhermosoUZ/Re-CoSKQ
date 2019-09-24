import logging
import os

import word2vec


def load_word2vec_model(file_name='model.bin'):
    """
    Loads a word2vec model given a file name from inside the project directory.
    :param file_name: The name of the file
    :return: The word2vec model
    """
    logger = logging.getLogger(__file__ + '.load_word2vec_model')
    model_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../../' + file_name)
    logger.debug('loading model from path {}'.format(model_path))
    try:
        model = word2vec.load(model_path)
    except:
        logger.error('Could not load model from path {}'.format(model_path))
        raise ValueError('Could not load model from path {}'.format(model_path))
    return model
