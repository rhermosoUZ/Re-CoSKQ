# Re-CoSKQ

Re-CoSKQ aims to provide recommendations based on the CoSKQ algorithm.

CoSKQ intents to retrieve best-cost subsets from a pool of possibilities given a query.
The query and every element in the set of data consist of physical coordinates and a set of keywords.
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
These metrics are used by CostFunctions.

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

### Simple Code Example

```python
from src.model.keyword_coordinate import KeywordCoordinate
from src.metrics.distance_metrics import manhattan_distance, euclidean_distance
from src.metrics.similarity_metrics import separated_cosine_similarity, combined_cosine_similarity
from src.utils.logging_utils import solution_list_comprehension
from src.utils.data_generator import DataGenerator
from src.costfunctions.type1 import Type1
from src.costfunctions.type2 import Type2
from src.solvers.naive_solver import NaiveSolver
from src.evaluator import Evaluator

# Evaluator, instantiate it first for logging purposes
ev = Evaluator()

# Create or load the data
query = KeywordCoordinate(0, 0, ['rest'])
kwc1 = KeywordCoordinate(2, 1, ['family'])
kwc2 = KeywordCoordinate(1, 2, ['food'])
kwc3 = KeywordCoordinate(2, 2, ['outdoor'])
data = [kwc1, kwc2, kwc3]
possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
dg = DataGenerator(possible_keywords)
gen_data = dg.generate(10)

# Define the cost functions
cf = Type1(manhattan_distance, separated_cosine_similarity, 0.2, 0.1, 0.7, disable_thresholds=False)
cf2 = Type2(euclidean_distance, combined_cosine_similarity, 0.2, 0.1, 0.7, disable_thresholds=True)

# Choose which solver to use
ns = NaiveSolver(query, data, cf, result_length=5)
ns2 = NaiveSolver(query, gen_data, cf2, result_length=5)

# Add Solvers to evaluator
ev.add_solver(ns)
ev.add_solver(ns2)

# Run evaluator and fetch results
ev.evaluate()
results = ev.get_results()
print(solution_list_comprehension(results))
```

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