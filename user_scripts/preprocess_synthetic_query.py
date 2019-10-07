from src.utils.data_generator import DataGenerator
from src.utils.data_handler import write_pickle

if __name__ == '__main__':
    # Config
    data_target_name = 'synthetic_query.pickle'
    possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
    file_allow_overwrite = False

    # Code
    dg = DataGenerator(possible_keywords)
    generated_query = dg.generate(1)
    write_pickle(generated_query, file_name=data_target_name, file_allow_overwrite=file_allow_overwrite)
