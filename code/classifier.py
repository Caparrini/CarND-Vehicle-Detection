from sklearn import svm
from sklearn.model_selection import GridSearchCV
import feature_extraction
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import joblib
import os
import numpy as np
import tools as utils
import itertools
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

def get_classifier(features, labels):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters,n_jobs=-1, verbose=5)
    clf.fit(features, labels)
    return clf

def test_classifier(clf):
    features_test, labels_test = feature_extraction.get_test()
    labels_predicted = clf.predict(features_test)
    acc = accuracy_score(labels_test, labels_predicted)
    cm = confusion_matrix(labels_test, labels_predicted)
    logging.info("Accuracy: {0}".format(acc))
    logging.info("Confusion matrix: {0}".format(cm))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function plots a confusion matrix

    :param numpy.array cm: Confusion matrix
    :param list classes: List of classes
    :param boolean normalize: True to normalize
    :param str title: Title of the plot
    :param cmap: Colours
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=sum(cm[0][:]))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).astype('int')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def KFoldCrossValidation(features, labels, report_folder, clf):
    '''
    Generates a report using KFold cross validation.
    It generate train/test confusion matrix for each kfold, a final kfold with all the test splits
    and a report.txt with metrics and other data.

    :param pandas.DataFrame df: DataFrame with the dataset
    :param str report_folder: folder where save pics and report
    :param clf: classifier with methods fit, score and predict sklearn styled
    :return: clf trained with all the data
    '''
    class_list = sorted(list(set(labels)))

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Report file with useful information
    if (os.path.isdir(report_folder)):

        logging.warning("The directory %s already exist", report_folder)

    else:

        logging.info("Creating directory %s", report_folder)

        os.mkdir(report_folder, 0o0755)


    report = open(os.path.join(report_folder,"report.txt"), "w")

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):

        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier with 80% of samples
        clf.fit(features_train, labels_train)
        # And predict with the other 20%
        accuracies_kfold.append(clf.score(features_test, labels_test))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test)

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1

    print(accuracies_kfold)
    print("\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(np.std(accuracies_kfold)) + "\n")
    report.write("Accuracies: " + str(accuracies_kfold) + "\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(
        np.std(accuracies_kfold)) + "\n")

    # Confusion matrix with all the predicted classes
    cm_kfold_total = confusion_matrix(labels_kfold, labels_kfold_predicted)

    # Get current size and making it bigger
    #fig_size = plt.rcParams["figure.figsize"]

    # Set figure according with the number of classes
    #size = len(class_list) - len(class_list)*30/100
    #fig_size[0] = size
    #fig_size[1] = size
    #plt.rcParams["figure.figsize"] = fig_size


    plt.figure()
    plot_confusion_matrix(cm_kfold_total, class_list, False, "Full test Confusion")
    plt.savefig(os.path.join(report_folder,"cmkfolds.pdf"))

    cmm = utils.ConfusionMatrixUtils(cm_kfold_total, class_list)
    report.write(cmm.report() + "\n\n")

    joblib.dump(cmm,os.path.join(report_folder,"cmm"))
    joblib.dump(cmm.cmmToGraph(),os.path.join(report_folder,"cmgraph"))

    clf.fit(features, labels)

    return clf

def KFoldAccuracy(features, labels, clf, n_splits=5, random_state=None):
    '''
    Computes KFold cross validation accuracy using n_splits folds over the data in the pandas.DataFrame given.
    Uses an stratified KFold with the random_state specified.

    :param df: pandas.DataFrame where is the data for train/test splits
    :param clf: classifier with methods fit, predict and score
    :param n_splits: number of splits
    :param random_state: random state seed
    :return: mean accuracy, std
    '''

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):
        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier
        clf.fit(features_train, labels_train)
        accuracies_kfold.append(clf.score(features_test, labels_test))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test)

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1

    meanAccuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)

    return meanAccuracy, std