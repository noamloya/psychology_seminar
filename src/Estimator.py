import src.DataSet as DataSet
import tensorflow as tf

DATA_PATH = '../resources/AllResults.xls'
BATCH_SIZE = 128

def main():
    train_sum = 0
    test_sum = 0
    iters = 5
    steps_arg = 1000
    is_stratified_sampling = True
    for k in range(0, iters):
        # Fetch the data
        (train_x, train_y), (test_x, test_y) = DataSet.load_data(DATA_PATH, is_stratified_sampling=is_stratified_sampling)

        # Feature columns describe how to use the input.
        my_feature_columns = []
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Build linear classifier.

        # --- List of possible classifiers.

        # This could be used with the default optimizer, 'Adam' or 'Adagrad'.
        classifier = tf.estimator.LinearClassifier(feature_columns=my_feature_columns, n_classes=2,
                                                   optimizer='SGD')

        # This could be used with various initial learning rates: if one isn't good enough, we can divide it by 10.
        #global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        #classifier = tf.estimator.LinearClassifier(feature_columns=my_feature_columns, n_classes=2,
        #                                           optimizer=tf.train.GradientDescentOptimizer(
        #                                               learning_rate=0.01))
        # THIS ISN'T A GOOD IDEA
        # classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, n_classes=2,
        #                                         optimizer='SGD', hidden_units=[3], activation_fn=tf.nn.sigmoid)

        # Train the Model
        classifier.train(
            input_fn=lambda: DataSet.train_input_fn(train_x, train_y,
                                                    BATCH_SIZE), steps=steps_arg)

        train_predictions = classifier.predict(
            input_fn=lambda: DataSet.eval_input_fn(train_x,
                                                   labels=None,
                                                   batch_size=BATCH_SIZE))
        print("Results on Training Data:")
        train_sum += analyze_results(train_predictions, train_y)

        # Generate predictions from the model
        predictions = classifier.predict(
            input_fn=lambda: DataSet.eval_input_fn(test_x,
                                                   labels=None,
                                                   batch_size=BATCH_SIZE))

        print("Results on Test Data:")
        test_sum += analyze_results(predictions, test_y)
        # template = 'Prediction is "{}" ({:.1f}%), expected "{}"\n'

    print("=== Summary ===\n"
          "%d iterations\n%d steps per training\n"
          "Training Data- Average accuracy: %f"
          "\nTest data- Average accuracy: %f" %
          (iters, steps_arg, train_sum/iters, test_sum/iters))


def class_to_string(label):
    if label == 1:
        return "Success"
    else:
        return "Fail"


def analyze_results(predictions, labels):
    accuracy_cnt = 0
    true_positives = 0
    false_positives = 0

    prediction_vector = []

    for pred_dict, expec in zip(predictions, labels):
        class_id = pred_dict['class_ids'][0]
        prediction_vector.append(class_id)
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
    #print("%d (%.2f) right predictions.\n%d (%.2f) wrong predictions."
    #      % (accuracy_cnt, accuracy, error_cnt,
    #         error_cnt / test_size))

    precision = 0 if true_positives + false_positives == 0 else (
        float(true_positives) / (true_positives + false_positives))
    recall = float(true_positives) / positive_labels

    print("Accuracy %.2f, Precision: %.2f, Recall %.2f" % (accuracy, precision, recall))

    # ("PREDICTIONS:", prediction_vector)

    return accuracy


if __name__ == '__main__':
    main()
