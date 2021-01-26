import csv
import logging
import math
import os
import pickle
import typing

import json
import ast
import numpy as np
import pandas as pd
import re

from src.model.keyword_coordinate import KeywordCoordinate
from src.utils.typing_definitions import dataset_type, keyword_dataset_type
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

# Auxiliar functions
def reviews2OneString(reviews):
    lists = []
    result = []
    resultToReturn = []
    try:
        lists = ast.literal_eval(reviews) # Here we have a list of lists
        for list in lists:
            result.append(' '.join(list))
            
        resultToReturn = ' '.join(result)
    except:
        pass
    
    #for list in lists:
    #    print('list: ', list)
    #    if len(list) > 0:
    #        for element in list:
    #            keyword_list.append()
    # print('***** Review: ', reviews)
    # print('Result: ', result)
        
    return resultToReturn
    

def pre_process(text):
    
    try:
        # lowercase
        text=text.lower()

        #remove tags
        text=re.sub("","",text)

        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
    except:
        pass
    
    return text

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def get_topN_keywords(doc, N, tfidf_transformer, cv, feature_names):
    
    try:
        #generate tf-idf for the given document
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        #extract only the top n; n here is 10
        keywords=extract_topn_from_vector(feature_names,sorted_items,N)

        # now print the results
        #print("\n=====Doc=====")
        #print(doc)
        #print("\n===Keywords===")
        #for k in keywords:
        #    print(k,keywords[k])

        return keywords
    except:
        print('*************************************************', doc)

def load_word2vec_model(file_name='model.pickle'):
    """
    Loads a word2vec model given a file name from inside the project directory.
    :param file_name: The name of the file
    :return: The word2vec model
    """
    logger = logging.getLogger(__file__ + '.load_word2vec_model')
    model_path = os.path.abspath(os.path.abspath(os.path.dirname(__file__)) + '/../../files/' + file_name)
    logger.debug('loading model from path {}'.format(model_path))
    try:
        model = load_pickle(file_name)
    except:
        logger.error('Could not load model from path {}'.format(model_path))
        raise ValueError('Could not load model from path {}'.format(model_path))
    return model


def write_pickle(data, file_name: str, file_allow_overwrite: bool = False,
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
    file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../files/' + file_name)
    if file_allow_overwrite:
        mode = 'wb'
    else:
        mode = 'xb'
    logger.debug('file mode set to {}'.format(mode))
    with open(file_path, mode=mode) as file:
        logger.debug(
            'opened file {} and generating pickle dump of data {}'.format(file_path, data))
        pickle.dump(data, file, protocol=pickle_protocol_version)


def load_pickle(file_name: str, path_relative_to_project_root: bool = True):
    """
    Loads a pickle and returns the unpickled dataset.
    :param file_name: The name of the file
    :param path_relative_to_project_root: If the path can be assumed as relative to the project
    :return: The loaded dataset
    """
    logger = logging.getLogger(__name__)
    logger.debug('loading pickle. File {} using path relative {}'.format(file_name, path_relative_to_project_root))
    if path_relative_to_project_root:
        #print ('*****' + os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../files/' + file_name))
        file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../files/' + file_name)
    else:
        file_path = file_name
    with open(file_path, mode='rb') as file:
        dataset: dataset_type = pickle.load(file)
    return dataset


def load_csv(file_name: str, x_coordinate_index: int, y_coordinate_index: int, keywords_index: int,
             keywords_delimiter: str = ' ',
             max_read_length: int = -1, delimiter: str = ',', newline: str = '', quotechar: str = '"',
             path_relative_to_project_root: bool = True) -> dataset_type:
    """
    Loads a csv file.
    :param file_name: The file name of the csv file. The file is usually in the project folder. Otherwise use the path_relative_to_project_root flag.
    :param x_coordinate_index: The index of the x coordinates
    :param y_coordinate_index: The index of the y coordinates
    :param keywords_index: The index of the keywords
    :param keywords_delimiter: The delimiter of the keywords
    :param max_read_length: The maximum number of lines to read
    :param delimiter: The csv cell delimiter
    :param newline: The newline delimiter
    :param quotechar: The quotechar symbol
    :param path_relative_to_project_root: The flag if the file name is relative to the project folder
    :return: The dataset of the csv
    """
    df = pd.read_csv(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../files/' + file_name), delimiter = ';', error_bad_lines=False, encoding = "unicode_escape")
    
    # Calculates topN keywords using TF-IDF
    # Removes rows with NaN values
    df.dropna(inplace=True)
    reviews = df['keywords_all']
    
    
    df['keyword lists IDF'] = reviews.apply(lambda x: reviews2OneString(x))
    
    #remove POIs with no reviews or NaN values
    df = df[df['keyword lists IDF'].str.len() != 0]
    #df.dropna(inplace=True)
    
    
    df['keyword lists IDF'] = df['keyword lists IDF'].apply(lambda x:pre_process(x))
    
    docs = df['keyword lists IDF'].tolist()
    
    #print(df['keyword lists IDF'][0])
    # Let's compute IDF   
        #1. Create a vocabulary of words, 
        #2. Ignore words that appear in 85% of documents, 
        #3. Eliminate stop words
    cv=CountVectorizer(max_df=0.85,stop_words='english')
    word_count_vector=cv.fit_transform(docs)
    
    print(np.shape(word_count_vector))
    
    # Let's compute IDF (test = IDF dataset)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    # Computing TF-IDF and Extracting Keywords
    # Get the whole vocabulary (all reviews for all POIs) in a list
    docs = df['keyword lists IDF'].tolist()
    
    feature_names=cv.get_feature_names()
    
    df['Top-Keywords-TFIDF'] = reviews.apply(lambda x: get_topN_keywords(x, 10, tfidf_transformer, cv, feature_names))
    
    
    ###########################################
    
    dataset: dataset_type = []
    # max_read_length -= 1  # because the length doesn't start counting at 0
    # if path_relative_to_project_root:
    #     file_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../files/' + file_name)
    # else:
    #     file_path = file_name
    
    # with open(file_path, mode='rt', newline=newline, encoding='utf8') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
    #     for row in reader:
    #         try:
    #             print(row[2])
    #             current_POI_name = row[2]
    #             # print(row[x_coordinate_index])
    #             current_coordinate_x = float(row[x_coordinate_index])
    #             # print(current_coordinate_x)
    #             current_coordinate_y = float(row[y_coordinate_index])
    #             # print(current_coordinate_y)
    #         except:
    #             print('----- Failure -----')
    #             if max_read_length > 0:
    #                 max_read_length += 1
    #             continue
    #         raw_keyword_list = row[keywords_index].split(keywords_delimiter)
            
    for i in df.index:
        current_POI_name = df['name'][i]
        current_coordinate_x = float(df['lat'][i])
        current_coordinate_y = float(df['lng'][i])
        current_keywords: keyword_dataset_type = ' '.join(df['Top-Keywords-TFIDF'][i])
    
            # current_keywords: keyword_dataset_type = []
            # for keyword in raw_keyword_list:
            #     stripped_keyword = keyword.strip()
            #     if len(stripped_keyword) > 0:
            #         current_keywords.append(stripped_keyword)
            # current_keyword_coordinate = KeywordCoordinate(current_POI_name, current_coordinate_x, current_coordinate_y, current_keywords)
        current_keyword_coordinate = KeywordCoordinate(current_POI_name, current_coordinate_x, current_coordinate_y, current_keywords)
        dataset.append(current_keyword_coordinate)

    return dataset


def rebalance_subsets(subsets: typing.List, min_number_of_subsets) -> typing.List:
    """
    Rearranges the passed in subset given the wanted number of subsets.
    :param subsets: The Subsets
    :param min_number_of_subsets:  The wanted number of subsets
    :return: The rearranged list of subsets
    """
    length_subsets = len(subsets)
    rebalanced_subsets = []
    for offset in range(min_number_of_subsets):
        for index_counter in range(length_subsets // min_number_of_subsets + 1):
            current_index = offset + index_counter * min_number_of_subsets
            try:
                current_pick = subsets[current_index]
                rebalanced_subsets.append(current_pick)
            except IndexError:
                pass
    return rebalanced_subsets


def split_subsets(subsets, min_number_of_subsets: int, rebalance: bool = True) -> typing.List[typing.Tuple]:
    """
    Calculates the split subsets. This is done in preparation for multiprocessing.
    :param subsets: The subsets
    :param min_number_of_subsets: The minimum number for the targeted number of processes. This is usually the equal to the number of available CPU cores.
    :param rebalance: If the passed subsets should be rearranged to better distribute the workload among the processes.
    :return: A list with the split subsets. It has a length of scaling factor * number of processors
    """
    if rebalance:
        list_of_subsets = rebalance_subsets(subsets, min_number_of_subsets)
    else:
        list_of_subsets = subsets
    length_of_input_subsets = len(list_of_subsets)
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
        new_subset = tuple(list_of_subsets[start:end])
        result.append(new_subset)
    return result


def calculate_model_subset(query: KeywordCoordinate, data: dataset_type, model):
    """
    Calculates the required subset of word2vec model data. This can significantly decrease memory allocation overhead.
    :param query: The query
    :param data: The data
    :param model: The model
    :return: A model with only the required data
    """
    new_model = dict()
    keywords = set()
    
    for kw in query.keywords:
        keywords.add(kw)
    # for kwc in query:
    #     for kw in kwc.keywords:
    #         keywords.add(kw)
    for kwc in data:
        for kw in kwc.keywords:
            keywords.add(kw)
            
    i = 1
    num_keywords = 0;
    for kw in keywords:
        # print('------> ', kw)
        try:
            if kw[0].isalpha(): # Whatch out! It doesn't work with symbols like &/... SOL: check if text is alphabetic
                # print('***** ', kw[0])
                new_model[kw.lower()] = model[kw.lower()]
                num_keywords = num_keywords + 1
        except:
            i = i + 1
            continue
    print('Words included: ', num_keywords)
    print('Words left apart: ', i)
    return new_model
