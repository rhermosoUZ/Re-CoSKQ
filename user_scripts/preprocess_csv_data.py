from src.utils.data_handler import load_csv, write_pickle

if __name__ == '__main__':
    # Config
    csv_file_name = 'Paris_dataset.csv'
    data_target_name = 'data20_dataset.pickle'
    x_index = 6 # Starts in 0
    y_index = 7
    keyword_index = 2
    max_read_length = -1 #20   # -1 to disable
    keyword_delimiter = ' '
    csv_delimiter = ';'
    csv_quotechar = '"'
    file_allow_overwrite = True

    # Code
    print('Loading CSV', csv_file_name)
    data = load_csv(file_name=csv_file_name, x_coordinate_index=x_index, y_coordinate_index=y_index,
                    keywords_index=keyword_index, keywords_delimiter=keyword_delimiter, delimiter=csv_delimiter,
                    quotechar=csv_quotechar, max_read_length=max_read_length)
    if len(data) > 0:
        print('Example Datapoint:', data[0].coordinates.x, data[0].coordinates.y, data[0].keywords)
        write_pickle(data=data, file_name=data_target_name, file_allow_overwrite=file_allow_overwrite)
    else:
        print('Could not load any data.')
