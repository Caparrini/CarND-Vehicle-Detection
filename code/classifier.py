from sklearn import svm
from sklearn.model_selection import GridSearchCV
import feature_extraction
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd

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
