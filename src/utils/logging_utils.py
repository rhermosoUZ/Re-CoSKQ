from utils.types import dataset_type


def dataset_comprehension(dataset: dataset_type) -> str:
    result = '['
    for kwc in dataset:
        result += '({}, {}) ['.format(kwc.coordinates.x, kwc.coordinates.y)
        for keyword in kwc.keywords:
            result += '\'{}\', '.format(keyword)
        result = result[:-2]
        result += '], '
    result = result[:-2]
    result += ']'
    return result
