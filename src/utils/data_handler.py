import logging
import os
import pickle
import typing

import word2vec

from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import dataset_type


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


def write_pickle(data: dataset_type, file_name: str, file_allow_overwrite: bool = False,
                 file_only_overwrite_dot_pickle_files: bool = True,
                 pickle_protocol_version: int = 4) -> typing.NoReturn:
    """
    Writes a dataset to disk as pickle format.
    :param data: The dataset
    :param file_name: The name of the file
    :param file_allow_overwrite: If files are allowed to be overwritten
    :param file_only_overwrite_dot_pickle_files: If the name of the file has to end with .pickle
    :param pickle_protocol_version: The protocol version of the pickle format
    """
    logger = logging.getLogger(__name__)
    logger.debug('writing pickle for file {} with protocol verion {}'.format(file_name, pickle_protocol_version))
    if file_only_overwrite_dot_pickle_files == True and file_name[-7:] != '.pickle':
        logger.error('Cannot overwrite file not ending in .pickle in safe mode')
        raise ValueError('Cannot overwrite file not ending in .pickle in safe mode')
    file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../' + file_name)
    if file_allow_overwrite:
        mode = 'wb'
    else:
        mode = 'xb'
    logger.debug('file mode set to {}'.format(mode))
    with open(file_path, mode=mode) as file:
        logger.debug(
            'opened file {} and generating pickle dump of data {}'.format(file_path, dataset_comprehension(data)))
        pickle.dump(data, file, protocol=pickle_protocol_version)


def load_pickle(file_name, path_relative_to_project_root: bool = True) -> dataset_type:
    """
    Loads a pickle and returns the unpickled dataset.
    :param file_name: The name of the file
    :param path_relative_to_project_root: If the path can be assumed as relative to the project
    :return: The loaded dataset
    """
    logger = logging.getLogger(__name__)
    logger.debug('loading pickle. File {} using path relative {}'.format(file_name, path_relative_to_project_root))
    if path_relative_to_project_root:
        file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../' + file_name)
    else:
        file_path = file_name
    with open(file_path, mode='rb') as file:
        dataset: dataset_type = pickle.load(file)
    return dataset
