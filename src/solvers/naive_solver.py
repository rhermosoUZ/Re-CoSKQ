from __future__ import annotations

import concurrent.futures
import logging
import math
import copy
import multiprocessing as mp
import pandas as pd

from src.costfunctions.costfunction import CostFunction
from src.metrics.distance_metrics import normalize_data, denormalize_result_data, geographic_distance
from src.model.keyword_coordinate import KeywordCoordinate
from src.solvers.solver import Solver
from src.utils.data_handler import split_subsets
from src.utils.logging_utils import dataset_comprehension, result_list_comprehension
from src.utils.typing_definitions import dataset_type, solution_list


class NaiveSolver(Solver):
    
    list_of_subsets = []
    normalised_query = ''
    """
    The NaiveSolver does not use any kind of heuristic. It calculates the cost for every possibility and returns the best results.
    """
    # Change max_number_of_concurrent_processes TO mp.cpu_count() if running in Linux-based (Multiprocessing doesn't work on Windows 10)
    def __init__(self, query: KeywordCoordinate, data: dataset_type, cost_function: CostFunction,
                 normalize: bool = True, result_length: int = 10, max_subset_size: int = math.inf,
                 max_number_of_concurrent_processes: int = 5, rebalance_subsets: bool = True):
        """
        Constructs a new NaiveSolver object.
        :param query: The query for which to solve for
        :param data: The data for which to solve for
        :param cost_function: The cost function to determine subset costs
        :param normalize: If the data should be normalized before being processed. The data will be denormalized before being returned.
        :param result_length: The size of the results (Top-N)
        :param max_subset_size: The maximum size of any subset used to calculate the solution
        :param rebalance_subsets: If the passed subsets should be rearranged to better distribute the workload among the processes
        """
        logger = logging.getLogger(__name__)
        logger.debug('creating with query {}, data {}, cost function {}, normalization {} and result length {}'.format(query, dataset_comprehension(data), cost_function, normalize, result_length))
        super().__init__(query, data, cost_function, normalize, result_length, max_subset_size,
                         max_number_of_concurrent_processes, rebalance_subsets)
        
        if self.query.keywords[0] != '0': # Only for precalculate distance
            self.list_of_subsets, self.normalised_query = self.preprocess_input()
        
        logger.debug('created with query {}, data {}, cost function {}, normalization {} and result length {}'.format(self.query, dataset_comprehension(self.data), self.cost_function, self.normalize_data, self.result_length))

    def solve(self) -> solution_list:
        """
        Implements the solution algorithm.
        :return: A list with tuples. Every tuple contains a cost and the corresponding subset of KeywordCoordinates.
        """
        logger = logging.getLogger(__name__)
        logger.info('solving for query {} and dataset {} using cost function {} and result length {}'.format(self.query,
                                                                                                              dataset_comprehension(
                                                                                                                  self.data),
                                                                                                              self.cost_function,
                                                                                                               self.result_length))
       
        result_list: solution_list = []
        
        # UNCOMMENT FOR MULTIPROCESSING (DOES NOT WORK IN WINDOWS 10)
        # with concurrent.futures.ProcessPoolExecutor(
        #         max_workers=self.max_number_of_concurrent_processes) as executor:
        #     future_list = []
        #     for subsets in list_of_split_subsets:
        #         future = executor.submit(self.get_cost_for_subset, query, subsets)
        #         future_list.append(future)
        #     for future in future_list:
        #         for solution in future.result():
        #             result_list.append(solution)
        
        # ONE PROCESSOR VERSION
        result_list = []
 
        for subset in self.list_of_subsets: # list_of_split_subsets if multiprocessing enabled.
            # print(i)
            # i = i + 1
            solution = self.get_cost_for_subset(self.normalised_query, subset)
            result_list.append((solution, subset))
            
            
        # MULTIPROCESSOR VERSION
        # for future in future_list:
        #     for solution in future.result():
        #         result_list.append(solution)
        #########################
        
        result_list.sort(key=lambda x: x[0])
        result_list = result_list[:self.result_length]
        denormalized_result_list = denormalize_result_data(result_list, self.denormalize_max_x, self.denormalize_min_x,
                                                           self.denormalize_max_y, self.denormalize_min_y)
        logger.info('solved for {} with length {}'.format(result_list_comprehension(denormalized_result_list),
                                                          self.result_length))
        return denormalized_result_list

    def preprocess_input_precalculate_only(self):
        
        #  Calculates distances between any pair of locations (POIs)
        distances = []
        for kwc in self.data:
            distance_list = []
            # print(kwc.coordinates.x)
            # print(kwc.coordinates.y)
            distance_list.append(str(kwc.coordinates.x) + ',' + str(kwc.coordinates.y))
            for kwc2 in self.data:
                distance_list.append(geographic_distance(kwc.coordinates, kwc2.coordinates))
            distances.append(distance_list)
        
        # Calculates distance list from query location to any POI
        distances_to_query = []
        for kwc in self.data:
            distances_one_poi = []
            distances_one_poi.append(str(kwc.coordinates.x) + ',' + str(kwc.coordinates.y))
            distances_one_poi.append(geographic_distance(self.query.coordinates, kwc.coordinates))
            distances_to_query.append(distances_one_poi)
        
        geographic_distances = pd.DataFrame(distances)
        geographic_distances.set_index(0, inplace=True)
        geographic_distances.columns = geographic_distances.index.tolist()
        
        # Calculates distances to query location from every POI
        distances_to_query_df = pd.DataFrame(distances_to_query)
        distances_to_query_df.set_index(0, inplace=True)
        distances_to_query_df.columns = ['Distances2Query']
        # print('>>>>>>> Query location ', self.query.coordinates.x, ',', self.query.coordinates.y)

        # candidates_set = self.get_all_candidates_heuristic(self.data, distances_to_query_df)
        return self.get_all_candidates_heuristic(self.data, distances_to_query_df)

    
    def preprocess_input(self):
        
        #  Calculates distances between any pair of locations (POIs)
        distances = []
        for kwc in self.data:
            distance_list = []
            # print(kwc.coordinates.x)
            # print(kwc.coordinates.y)
            distance_list.append(str(kwc.coordinates.x) + ','+str(kwc.coordinates.y))
            for kwc2 in self.data:
                distance_list.append(geographic_distance(kwc.coordinates, kwc2.coordinates))
            distances.append(distance_list)
        
        # Calculates distance list from query location to any POI
        distances_to_query = []
        for kwc in self.data:
            distances_one_poi = []
            distances_one_poi.append(str(kwc.coordinates.x) + ',' + str(kwc.coordinates.y))
            distances_one_poi.append(geographic_distance(self.query.coordinates, kwc.coordinates))
            distances_to_query.append(distances_one_poi)
        
        geographic_distances = pd.DataFrame(distances)
        geographic_distances.set_index(0, inplace=True)
        geographic_distances.columns = geographic_distances.index.tolist()
        
        # Calculates distances to query location from every POI
        distances_to_query_df = pd.DataFrame(distances_to_query)
        distances_to_query_df.set_index(0, inplace=True)
        distances_to_query_df.columns = ['Distances2Query']
        
        
        
        # candidates_set = self.get_all_candidates_heuristic(self.data, distances_to_query_df)
        dataAux = self.get_all_candidates_heuristic(self.data, distances_to_query_df)
    
        # list_of_subsets = self.get_all_subsets(data)
        
        #  ONLY ONE PROCESSOR (NO BALANCING NEEDED)
        # list_of_split_subsets = list_of_subsets
        
        # dataAux = [x for x in self.data if (str(x.coordinates.x)+','+str(x.coordinates.y)) in candidates_set]
        
        if self.normalize_data:
            query, data, self.denormalize_max_x, self.denormalize_min_x, self.denormalize_max_y, self.denormalize_min_y = normalize_data(
                self.query, dataAux)
        else:
            query = self.query
            data = self.dataAux
            
        list_of_subsets = self.get_all_subsets(data)
        
        #  UNCOMMENT IF MULTIPROCESSING
        # list_of_split_subsets = split_subsets(list_of_subsets, self.max_number_of_concurrent_processes,
                                               # self.rebalance_subsets)

        print('List of subsets length: ', len(list_of_subsets))
    
        return list_of_subsets, query                                  

    def get_cost_for_subset(self, query, subset):
        """
        Calculates the costs of all the subsets for a given query and cost function.
        :param query: The query
        :param subsets: The list of subsets
        :param costfunction: The costfunction
        :return: A list of solutions. Each solution being a cost and the corresponding subset
        """
        # results: solution_list = []
        # print('***** subsets lenght: ', len(subset))
        # i = 1;
        # print('i = ', i)
        # i = i + 1
        
        denormalized_result = denormalize_result_data([(0.0, subset)], self.denormalize_max_x,
                                                      self.denormalize_min_x, self.denormalize_max_y,
                                                      self.denormalize_min_y)
        # print('Bottleneck 1')
        denormalized_subset = denormalized_result[0][1]
        current_result = self.cost_function.solve(query, subset, denormalized_subset)
        # print('Bottleneck 2')
        # results.append((current_result, subset))
        return current_result
