from src.utils.data_generator import DataGenerator

if __name__ == '__main__':
    # Config
    data_target_name = 'synthetic20_dataset.pickle'
    possible_keywords = ['family', 'food', 'outdoor', 'rest', 'indoor', 'sports', 'science', 'culture', 'history']
    dataset_size = 15
    file_allow_overwrite = False

    # Code
    dg = DataGenerator(possible_keywords)
    dg.generate_pickle(data_size=dataset_size, file_name=data_target_name, file_allow_overwrite=file_allow_overwrite)
