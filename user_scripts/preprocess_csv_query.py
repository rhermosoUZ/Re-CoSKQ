from src.utils.data_handler import load_csv, write_pickle

if __name__ == '__main__':
    # Config
    csv_file_name = 'user_queries.csv'
    data_target_name = 'data20_query.pickle'
    x_index = 0
    y_index = 1
    keyword_index = 2
    query_index = 22
    keyword_delimiter = ' '
    csv_delimiter = ';'
    csv_quotechar = '"'
    file_allow_overwrite = True

    # Code
    if query_index <= 0:
        print('The query row has to be positive.')
    else:
        print('Loading CSV', csv_file_name)
        data = load_csv(file_name=csv_file_name, x_coordinate_index=x_index, y_coordinate_index=y_index,
                        keywords_index=keyword_index, keywords_delimiter=keyword_delimiter, delimiter=csv_delimiter,
                        quotechar=csv_quotechar, max_read_length=query_index + 1)
        if len(data) > 0:
            print('Query Datapoint:', data[query_index - 1].coordinates.x, data[query_index - 1].coordinates.y,
                  data[query_index - 1].keywords)
            write_pickle(data=data[query_index - 1], file_name=data_target_name,
                         file_allow_overwrite=file_allow_overwrite)

        else:
            print('Could not load any data.')
