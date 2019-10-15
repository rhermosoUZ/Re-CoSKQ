# Re-CoSKQ

Re-CoSKQ aims to provide recommendations based on the CoSKQ algorithm.

CoSKQ intents to retrieve best-cost subsets from a pool of possibilities given a query.
The query and every element in the set of data consists of physical coordinates and a set of keywords.
In order to calculate the costs of a subset, CoSKQ calculates the physical distance between the query and the subset and also the physical distance between the entries of the subset.
These calculations are only carried out if the combined keywords of the subset cover the keywords of the query.

Plain CoSKQ is not suitable for Re-CoSKQ as it requires complete matches between the subset's keywords and the query's keywords.
Therefore, Re-CoSKQ calculates these best-cost subsets based on their physical distance to the query, the distance between the subset entries and the similarity between the query and subset keywords.
However, Re-CoSKQ does not strictly exclude subsets because of the keyword coverage.

## Project Setup

The Project consists of multiple classes and functions to accommodate the needs of Re-CoSKQ.
In general, it has been built with modularity and extensibility in mind.
New functionality can easily be added by inheriting from a base class or writing an independent function.
This new functionality can then be used by passing it as parameters into the framework.

### Distance and Similarity Metrics

These metrics define how physical distances and keyword similarities are calculated.
The physical distance metric is used for query-dataset and inter-dataset distances.
The keyword similarity metric is used for query-dataset similarities.
These metrics are used by the CostFunctions.

### CostFunctions

A CostFunction implements how the cost for a single subset should be calculated.
The CostFunction is frequently called by a Solver to calculate the costs for a given subsets.

### Solvers

A Solver implements how a set of data should be handled.
It estimates how the subsets are determined and how the resulting costs are handled.
Solvers also contain the query and data they will run on.
In general, a solver dictates how a single trial is executed.

### Evaluator

The Evaluator contains the logic to compare multiple Solvers.
It has a state and can be used interactively.

## Using the Application

Please refer to the user_scripts folder for code examples on how to use re-coskq.
Using this application can be split into three main categories:
 - Data Preprocessing
 - Precalculating Values
 - Running Evaluations

### Data Preprocessing

Data preprocessing usually only needs to be done once per dataset.
It involves reading a CSV file, generating synthetic data and generating the Word2Vec model.

#### CSV Data

CSV data is the preferred way of getting real data into this application.
During this process the CSV files are being read, 
turned into an application specific data structure and then saved as a pickle file.

Please refer to the following files inside the user_scripts folder:
 - preprocess_csv_data.py
 - preprocess_csv_query.py
 
 The difference between these two files is 
 that the first one reads the csv and  turns all read lines into a dataset, 
 while the second reads a single line of the dataset and turns it into a query.
 
 This has been tested with the geoname dataset from the [Chan et al., 2018] paper.
 
 The links to the datasets are the following:
 - US Hotels: www.allstays.com
 - US Geonames: geonames.usgs.gov
 

#### Synthetic Data

Synthetic data can be used to use the application if no real data is available or said data is insufficient.
During this process data is generated based on passed parameters such as the coordinate limits and possible keywords.
Once the synthetic data has been generated, it is turned into an application specific data structure and saved as a pickle file.

Please refer to the following files inside the user_scripts folder:
 - preprocess_synthetic_data.py
 - preprocess_synthetic_query.py
 
 The difference between these two files can be described just as for the CSV data.
 Once generates a dataset and the other a query.

#### Word2Vec Model

The Word2Vec model is required if the keyword similarity is to be determined using the word2vec similarity.
During this process a word2vec binary is generated from a body of text.
This binary is then converted into a pickle file.
This pickle file is then stripped of unnecessary information to speed up memory allocations.

Please refer to the following files inside the user_scripts folder:
 - word2vec_model_generator.py
 - word2vec_model_pickler.py
 - word2vec_model_to_data and query_adapter.py
 
 The generator file generates a binary model from a body of text.
 The pickler turns this binary file into a pickle file and strips unnecessary methods.
 And finally, the adapter removes unnecessary words from vocabulary of the model.
 
 The word2vec documentation and a dataset to train the model can be found at: https://pypi.org/project/word2vec/
 
### Precalulating Values

Precalucating values makes sense if there are multiple trials with different parameters but the same dataset.
It also makes sense if only a part of the data or query changes.
For example, the inter-dataset distances are going to stay the same if the query changes.
The query-dataset and the inter-dataset distances are going to stay the same if only keywords change.
And, the keyword similarities are going to stay the same if only the coordinates of the dataset or the query change.

Please refer to the following files inside the user_scripts folder:
 - precalculate_inter_dataset_distances.py
 - precalculate_query_dataset_distances.py
 - precalculate_query_dataset_keyword_similarities.py
 - precalculate_query_dataset_keyword_similarities_word2vec.py

All the above scripts do exactly as their names suggest.
The difference between the regular keyword similarity and the Word2Vec keyword similarity being 
the structure of the script due to the additional loading of the model.
It generally makes sense to precalculate and reuse results wherever possible.

### Running Evaluations

The evaluation is where it all comes together.
First, load the dataset and query.
Then, load the precalculated values and models.
Once that is done, set up the cost functions with the precalculated values and models.
After that, pass in the data, query and costfunctions into the chosen solvers.
And finally add everything to the evaluator and run the evaluation.

Please refer to the following file inside the user_scripts folder:
 - evaluate.py

## Running the Tests

All the included unit tests can be run using the following steps:

 - Start in the test directory of the project (usually the re-coskq/test folder)
 - Run: python -m unittest discover --top-level-directory ..
 
## Building the Documentation

Requires make and sphinx to be installed.
First the .rst files need to be generated from the .py source files.
Once they are generated, the documentation can be built from the .rst files.
Possible targets include: html, singlehtml and latexpdf.

Take the following steps to generate the documentation:
 - Start from the root of the project (usually the re-coskq folder)
 - Generate the .rst files: sphinx-apidoc --implicit-namespaces --force -o docs/ src/
 - Go to the docs/ directory: cd docs/
 - (If something went wrong previously: make clean)
 - Build the target: make html