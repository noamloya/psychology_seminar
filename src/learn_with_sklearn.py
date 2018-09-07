import src.DataSet as DataSet
import numpy as np
import sklearn.linear_model as lin

DATA_PATH = '../resources/AllResults.xls'
BATCH_SIZE = 128

def main():
    train_sum = 0
    test_sum = 0
    iters = 5
    steps_arg = 10000
    is_stratified_sampling = True
    test_accs, test_prcs, test_recs = [], [], []
    train_accs, train_prcs, train_recs = [], [], []
    for k in range(0, iters):
        # Fetch the data
        (train_x, train_y), (test_x, test_y) = DataSet.load_data(DATA_PATH, is_stratified_sampling=is_stratified_sampling)

        classifier = lin.LinearRegression()
        # Train the Model
        classifier.fit(train_x, train_y)

        print("Training score:", classifier.score(train_x, train_y))
        train_predictions = classifier.predict(train_x)
        print("Results on Training Data:")
        train_acc, train_prc, train_rec = analyze_results(train_predictions, train_y)
        train_accs.append(train_acc)
        train_prcs.append(train_prc)
        train_recs.append(train_rec)


        # Generate predictions from the model
        predictions = classifier.predict(test_x)

        print("Results on Test Data:")
        test_acc, test_prc, test_rec = analyze_results(predictions, test_y)
        test_accs.append(test_acc)
        test_prcs.append(test_prc)
        test_recs.append(test_rec)
        # template = 'Prediction is "{}" ({:.1f}%), expected "{}"\n'

    train_accs = np.array(train_accs)
    train_prcs = np.array(train_prcs)
    train_recs = np.array(train_recs)
    test_accs = np.array(test_accs)
    test_prcs = np.array(test_prcs)
    test_recs = np.array(test_recs)

    print("=== Summary ===")
    print("%d iterations" % iters)
    print("%d steps per training" % steps_arg)
    print("Training Data- Accuracy: Average: %f, Std: %f" % (np.mean(train_accs), np.std(train_accs)))
    print("Training Data- Precision: Average: %f, Std: %f" % (np.mean(train_prcs), np.std(train_prcs)))
    print("Training Data- Recall: Average: %f, Std: %f" % (np.mean(train_recs), np.std(train_recs)))
    print("Test Data- Accuracy: Average: %f, Std: %f" % (np.mean(test_accs), np.std(test_accs)))
    print("Test Data- Precision: Average: %f, Std: %f" % (np.mean(test_prcs), np.std(test_prcs)))
    print("Test Data- Recall: Average: %f, Std: %f" % (np.mean(test_recs), np.std(test_recs)))




def class_to_string(label):
    if label == 1:
        return "Success"
    else:
        return "Fail"


def analyze_results(predictions, labels):
    predictions = np.round(predictions)
    accuracy_cnt = 0
    true_positives = 0
    false_positives = 0

    prediction_vector = []

    accuracy_cnt = np.sum(predictions == labels)
    true_positives = np.sum((predictions - labels) > 0)
    false_positives = np.sum(1 - np.absolute(predictions != labels))

    test_size = len(labels)
    positive_labels = sum(labels > 0)

    accuracy = accuracy_cnt / test_size

    precision = 0 if true_positives + false_positives == 0 else (
        float(true_positives) / (true_positives + false_positives))
    recall = float(true_positives) / positive_labels

    print("Accuracy %.2f, Precision: %.2f, Recall %.2f" % (accuracy, precision, recall))

    # ("PREDICTIONS:", prediction_vector)

    return accuracy, precision, recall


if __name__ == '__main__':
    main()
