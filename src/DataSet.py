import pandas as pd
import tensorflow as tf
import random
import math

CSV_COLUMN_NAMES = ['Tas', 'Stai', 'Stroop', 'Adap', 'S_Min']
SUCCESS_INDEX = [0, 1]  # defined by 65 percentile of S MIN
TEST_PERCENT = 0.2


def load_data(data_path, y_name='S_Min', is_stratified_sampling=False):
    """Given a dataset file path, splits the data randomly into training set and testing set.
    :returns Returns the data set as (train_x, train_y), (test_x, test_y)."""

    file_obj = pd.read_excel(data_path, names=CSV_COLUMN_NAMES, header=0)

    train, test = create_train_and_test(file_obj, y_name, is_stratified_sampling)
    test_x, test_y = test, test.pop(y_name)
    train_x, train_y = train, train.pop(y_name)
    # print("after random split: number of success(1) in test labels is: %d out of %d\n" % (sum(test_y), len(test_y)))
    # print("after random split: number of success(1) in train labels is: %d out of %d\n" % (sum(train_y), len(train_y)))

    return (train_x, train_y), (test_x, test_y)


def get_test_size(sample_size):
    return math.ceil(sample_size * TEST_PERCENT)


def create_train_and_test(file_obj, label_name, is_stratified_sampling):
    if is_stratified_sampling:
        positive_samples = file_obj.loc[file_obj[label_name] == 1]
        positive_test_set = positive_samples.sample(frac=TEST_PERCENT)
        # Gets the rows that aren't in the test set.
        positive_training_set = positive_samples[~positive_samples.isin(positive_test_set)].dropna()

        negative_samples = file_obj.loc[file_obj[label_name] == 0]
        negative_test_set = negative_samples.sample(frac=TEST_PERCENT)
        negative_training_set = negative_samples[~negative_samples.isin(negative_test_set)].dropna()

        train = pd.concat([positive_training_set, negative_training_set])
        test = pd.concat([positive_test_set, negative_test_set])
    else:
        sample_size = file_obj.shape[0]
        test_size = get_test_size(sample_size)
        test_lines = random.sample(range(0, sample_size), test_size)
        test = file_obj.iloc[test_lines, :]
        train = file_obj.drop(test_lines, axis=0)

    return train, test


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    data_set = data_set.shuffle(1000).repeat().batch(batch_size)
    return data_set


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
