import DataSet
import numpy as np
import sklearn.linear_model as lin
from sklearn import metrics, svm
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = '../resources/Smin_v02.xls'
BATCH_SIZE = 128
NUM_ITERATIONS = 1
is_stratified_sampling = True


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def run_and_evaluate_model(model, model_name, iters, is_stratified_sampling):
    test_accs, test_prcs, test_recs = [], [], []
    train_accs, train_prcs, train_recs = [], [], []
    for k in range(iters):
        # Fetch the data
        (train_x, train_y), (test_x, test_y) = DataSet.load_data(DATA_PATH, is_stratified_sampling=is_stratified_sampling)

        classifier = model

        # Train the Model
        classifier.fit(train_x, train_y)
        print("Training score:", classifier.score(train_x, train_y))

        train_predictions = classifier.predict(train_x)
        # print("Results on Training Data:")
        train_acc, train_prc, train_rec = analyze_results(train_predictions, train_y, model_name + '_train')
        train_accs.append(train_acc)
        train_prcs.append(train_prc)
        train_recs.append(train_rec)

        # Generate predictions from the model
        predictions = classifier.predict(test_x)

        # print("Results on Test Data:")
        test_acc, test_prc, test_rec = analyze_results(predictions, test_y, model_name + '_test')
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
    print("Training Data- Accuracy: Average: %f, Std: %f" % (np.mean(train_accs), np.std(train_accs)))
    print("Training Data- Precision: Average: %f, Std: %f" % (np.mean(train_prcs), np.std(train_prcs)))
    print("Training Data- Recall: Average: %f, Std: %f" % (np.mean(train_recs), np.std(train_recs)))
    print("Test Data- Accuracy: Average: %f, Std: %f" % (np.mean(test_accs), np.std(test_accs)))
    print("Test Data- Precision: Average: %f, Std: %f" % (np.mean(test_prcs), np.std(test_prcs)))
    print("Test Data- Recall: Average: %f, Std: %f" % (np.mean(test_recs), np.std(test_recs)))


def main():
    print('=====================================')
    print(DataSet.CSV_COLUMN_NAMES)
    print('=====================================')
    models = {
        'Linear Regression': lin.LinearRegression(),
        'Ridge Regression': lin.Ridge(),
        'Logistic Regression': lin.LogisticRegression(),
        'Linear SVM': svm.LinearSVC(),
        'SVM With RBF Kernel': svm.SVC(),
        'Random Forest':  RandomForestRegressor(n_jobs=-1, n_estimators=100),
        'Random Forest Limited Depth': RandomForestRegressor(n_jobs=-1, n_estimators=10, max_depth=2),
    }

    for model_name in models:
        print("======== Model: %s ========" % model_name)
        run_and_evaluate_model(models[model_name], model_name, NUM_ITERATIONS, is_stratified_sampling)


def class_to_string(label):
    if label == 1:
        return "Success"
    else:
        return "Fail"


def analyze_results(predictions, labels, fig_name):
    predictions = np.round(predictions)
    accuracy_cnt = np.sum(predictions == labels)
    true_positives = np.sum(np.logical_and(predictions == 1, labels == 1))
    false_positives = np.sum(np.sum(np.logical_and(predictions == 1, labels == 0)))

    test_size = len(labels)
    positive_labels = sum(labels > 0)

    accuracy = accuracy_cnt / test_size

    precision = 0 if true_positives + false_positives == 0 else (
        float(true_positives) / (true_positives + false_positives))
    recall = float(true_positives) / positive_labels

    # print("Accuracy %.2f, Precision: %.2f, Recall %.2f" % (accuracy, precision, recall))

    cnf_matrix = metrics.confusion_matrix(labels, predictions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    if NUM_ITERATIONS == 1:
        fig = plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['Unsuccessful', 'Successful'],
                              title='Confusion matrix, Normalized', normalize=True)
        fig.savefig('./Confustion_matrix_%s.jpg' % fig_name)
        fig.clf()
    return accuracy, precision, recall


if __name__ == '__main__':
    main()
