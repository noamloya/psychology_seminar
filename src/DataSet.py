import pandas as pd
import tensorflow as tf
import random
import math

CSV_COLUMN_NAMES = ['Tas', 'Stai', 'SuccessIndex']
SUCCESS_INDEX = [0, 1]  # defined by 65 percentile of S MIN
TEST_PERCENT = 0.2


def load_data(data_path, y_name='SuccessIndex'):
    """Given a dataset file path, splits the data randomly into training set and testing set.
    :returns Returns the data set as (train_x, train_y), (test_x, test_y)."""

    file_obj = pd.read_excel(data_path, names=CSV_COLUMN_NAMES, header=0)
    sample_size = file_obj.shape[0]
    test_lines = select_test_set(sample_size)
    test = file_obj.iloc[test_lines, :]
    test_x, test_y = test, test.pop(y_name)

    train = file_obj.drop(test_lines, axis=0)
    train_x, train_y = train, train.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


def select_test_set(sample_size):
    test_size = math.ceil(sample_size * TEST_PERCENT)
    test_lines = random.sample(range(0, sample_size), test_size)
    return test_lines


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return data_set.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Data set.
    data_set = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    data_set = data_set.batch(batch_size)

    # Return the data set.
    return data_set
