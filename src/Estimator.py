import src.DataSet as DataSet
import tensorflow as tf

DATA_PATH = 'resources/AllResults.xls'
BATCH_SIZE = 128


def main():
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = DataSet.load_data(DATA_PATH)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Build linear classifier.
    classifier = tf.estimator.LinearClassifier(feature_columns=my_feature_columns, n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda: DataSet.train_input_fn(train_x, train_y,
                                                BATCH_SIZE), steps=1000)

    # Generate predictions from the model
    predictions = classifier.predict(
        input_fn=lambda: DataSet.eval_input_fn(test_x,
                                               labels=None,
                                               batch_size=BATCH_SIZE))

    template = 'Prediction is "{}" ({:.1f}%), expected "{}"\n'

    accuracy_cnt = 0

    for pred_dict, expec in zip(predictions, test_y):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        if class_id == expec:
            accuracy_cnt += 1

        print(template.format(class_to_string(DataSet.SUCCESS_INDEX[class_id]),
                              100 * probability, class_to_string(expec)))

    test_size = len(test_y)
    error_cnt = test_size - accuracy_cnt
    print("%d (%.2f) right predictions.\n%d (%.2f) wrong predictions."
          % (accuracy_cnt, accuracy_cnt / test_size, error_cnt,
             error_cnt / test_size))


def class_to_string(label):
    if label == 1:
        return "Success"
    else:
        "Fail"


if __name__ == '__main__':
    main()
