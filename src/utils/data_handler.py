import csv
import logging
import math
import multiprocessing as mp
import os
import pickle
import typing

import word2vec

from src.model.keyword_coordinate import KeywordCoordinate
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


def load_csv(file_name, x_coordinate_index, y_coordinate_index, keywords_index, keywords_delimiter=' ',
             max_read_length=-1, delimiter=',', newline='', quotechar='"',
             path_relative_to_project_root: bool = True) -> dataset_type:
    dataset: dataset_type = []
    max_read_length -= 1  # because the length doesn't start counting at 0
    if path_relative_to_project_root:
        file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../' + file_name)
    else:
        file_path = file_name
    with open(file_path, mode='rt', newline=newline) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            try:
                current_coordinate_x = float(row[x_coordinate_index])
                current_coordinate_y = float(row[y_coordinate_index])
            except:
                if max_read_length > 0:
                    max_read_length += 1
                continue
            raw_keyword_list = row[keywords_index].split(keywords_delimiter)
            current_keywords: typing.List[str] = []
            for keyword in raw_keyword_list:
                stripped_keyword = keyword.strip()
                if len(stripped_keyword) > 0:
                    current_keywords.append(stripped_keyword)
            current_keyword_coordinate = KeywordCoordinate(current_coordinate_x, current_coordinate_y, current_keywords)
            dataset.append(current_keyword_coordinate)
            if len(dataset) == max_read_length:
                return dataset
    return dataset


def split_subsets(subsets, scaling_factor_number_of_processes: int = 2) -> typing.List[typing.Tuple]:
    min_number_of_subsets = mp.cpu_count() * scaling_factor_number_of_processes
    length_of_input_subsets = len(subsets)
    length_per_subset = math.floor(length_of_input_subsets / min_number_of_subsets)
    if length_per_subset == 0:
        length_per_subset = 1
    mod_length_jobs = length_of_input_subsets % length_per_subset
    if mod_length_jobs == 0:
        total_number_of_subsets = length_of_input_subsets // length_per_subset
    else:
        total_number_of_subsets = (length_of_input_subsets // length_per_subset) + 1
    result: typing.List[typing.Tuple] = []
    for count in range(total_number_of_subsets):
        start = count * length_per_subset
        end = (count + 1) * length_per_subset
        new_subset = tuple(subsets[start:end])
        result.append(new_subset)
    return result
