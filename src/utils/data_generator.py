import logging
import os
import pickle
import random
import typing

from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.logging_utils import dataset_comprehension
from src.utils.typing_definitions import dataset_type


class DataGenerator:
    """
    The DataGenerator class offers convenience functions to generate data and save it to disk.
    """
    def __init__(self, possible_keywords: typing.List[str], keywords_min: int = 1, keywords_max: int = 5, physical_min_x: float = 0.0, physical_max_x: float = 100.0, physical_min_y: float = 0.0, physical_max_y: float = 100.0):
        """
        Instantiates a new DataGenerator object.
        :param possible_keywords: A list of all the possible keywords
        :param keywords_min: The minimum of keywords per KeywordCoordinate
        :param keywords_max: The maximum of keywords per KeywordCoordinate
        :param physical_min_x: The minimum x value for the KeywordCoordinates
        :param physical_max_x: The maximum x value for the KeywordCoordinates
        :param physical_min_y: The minimum y value for the KeywordCoordinates
        :param physical_max_y: The maximum y value for the KeywordCoordinates
        """
        logger = logging.getLogger(__name__)
        logger.debug('initializing DataGenerator using possible keywords {}, keywords_min {}, keywords_max {}, physical_min_x {}, physical_max_x {}, physical_min_y {} and physical_max_y {}'.format(possible_keywords, keywords_min, keywords_max, physical_min_x, physical_max_x, physical_min_y, physical_max_y))
        self.possible_keywords = possible_keywords
        self.keywords_min = keywords_min
        self.keywords_max = keywords_max
        self.physical_min_x = physical_min_x
        self.physical_max_x = physical_max_x
        self.physical_min_y = physical_min_y
        self.physical_max_y = physical_max_y

    def generate(self, data_size: int) -> dataset_type:
        """
        Generates a dataset with a given size.
        :param data_size: The size of the dataset
        :return: The dataset
        """
        logger = logging.getLogger(__name__)
        logger.debug('generating dataset of size {}'.format(data_size))
        dataset: dataset_type = []
        for data_counter in range(data_size):
            possible_keywords_copy = self.possible_keywords.copy()
            current_keywords: typing.List[str] = []
            current_x = random.randint(self.physical_min_x, self.physical_max_x)
            current_y = random.randint(self.physical_min_y, self.physical_max_y)
            number_of_keywords = random.randint(self.keywords_min, self.keywords_max)
            for kw_counter in range(number_of_keywords):
                try:
                    current_keyword = random.choice(possible_keywords_copy)
                except IndexError:
                    break
                possible_keywords_copy.remove(current_keyword)
                current_keywords.append(current_keyword)
            new_entry = KeywordCoordinate(current_x, current_y, current_keywords)
            dataset.append(new_entry)
        logger.debug('generated dataset {}'.format(dataset_comprehension(dataset)))
        return dataset

    def write_pickle(self, data: dataset_type, file_name: str, file_allow_overwrite: bool = False, file_only_overwrite_dot_pickle_files: bool = True, pickle_protocol_version: int = 4) -> typing.NoReturn:
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
            logger.debug('opened file {} and generating pickle dump of data {}'.format(file_path, dataset_comprehension(data)))
            pickle.dump(data, file, protocol=pickle_protocol_version)

    def generate_pickle(self, data_size: int, file_name: str, file_allow_overwrite: bool = False, file_only_overwrite_dot_pickle_files: bool = True, pickle_protocol_version: int = 4) -> dataset_type:
        """
        Generates a new dataset, writes it as pickle and returns the generated dataset.
        :param data_size: The dataset
        :param file_name: The name of the file
        :param file_allow_overwrite: If files are allowed to be overwritten
        :param file_only_overwrite_dot_pickle_files: If the name of the file has to end with .pickle
        :param pickle_protocol_version: The protocol version of the pickle format
        :return: The generated data which has been written to disk
        """
        logger = logging.getLogger(__name__)
        logger.debug('generating dataset of size {}'.format(data_size))
        data = self.generate(data_size)
        logger.debug('generated dataset {}'.format(dataset_comprehension(data)))
        self.write_pickle(data, file_name, file_allow_overwrite, file_only_overwrite_dot_pickle_files, pickle_protocol_version)
        return data

    def load_pickle(self, file_name, path_relative_to_project_root:bool = True) -> dataset_type:
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