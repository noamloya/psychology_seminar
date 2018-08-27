import DataSet
import tensorflow as tf

DATA_PATH = '../resources/AllResults.xls'
BATCH_SIZE = 128


def main():
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = DataSet.load_data(DATA_PATH)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Build linear classifier.
    classifier = tf.estimator.LinearClassifier(feature_columns=my_feature_columns, n_classes=2,
                                               optimizer='SGD')

    # Train the Model.
    classifier.train(
        input_fn=lambda: DataSet.train_input_fn(train_x, train_y,
                                                BATCH_SIZE), steps=1000)

    train_predictions = classifier.predict(
        input_fn=lambda: DataSet.eval_input_fn(train_x,
                                               labels=None,
                                               batch_size=BATCH_SIZE))

    print("Results on Training Data:")
    analyze_results(train_predictions, train_y)

    # Generate predictions from the model
    predictions = classifier.predict(
        input_fn=lambda: DataSet.eval_input_fn(test_x,
                                               labels=None,
                                               batch_size=BATCH_SIZE))

    print("Results on Test Data:")
    analyze_results(predictions, test_y)
    # template = 'Prediction is "{}" ({:.1f}%), expected "{}"\n'
    #


def class_to_string(label):
    if label == 1:
        return "Success"
    else:
        return "Fail"


def analyze_results(predictions, labels):
    accuracy_cnt = 0
    true_positives = 0
    false_positives = 0

    for pred_dict, expec in zip(predictions, labels):
        class_id = pred_dict['class_ids'][0]

        if class_id == expec:
            accuracy_cnt += 1

        if expec > 0 and class_id == expec:
            true_positives += 1
        elif expec <= 0 and class_id != expec:
            false_positives += 1

    test_size = len(labels)
    positive_labels = sum(labels > 0)

    error_cnt = test_size - accuracy_cnt
    accuracy = accuracy_cnt / test_size
    print("%d (%.2f) right predictions.\n%d (%.2f) wrong predictions."
          % (accuracy_cnt, accuracy, error_cnt,
             error_cnt / test_size))

    precision = 0 if true_positives + false_positives == 0 else (
        float(true_positives) / (true_positives + false_positives))
    recall = float(true_positives) / positive_labels

    print("Accuracy %.2f, Precision: %.2f, Recall %.2f" % (accuracy, precision, recall))


if __name__ == '__main__':
    main()
