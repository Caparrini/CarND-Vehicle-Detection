import cv2
import numpy as np
import utils
import classifier
import feature_extraction
import optimize
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import logging
import os
logging.getLogger().setLevel(level=logging.INFO)


#Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

#Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

#Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

#Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

#Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
## and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

#Estimate a bounding box for vehicles detected.

def best_extratrees(features, labels, pop=3, gen=2):
    opt = optimize.ExtraTreesOptimizer(features, labels)
    clf = opt.optimizeClf(pop,gen)
    return clf

def main():
    logging.info("Extract features from dataset")
    if (os.path.exists(os.path.join("Report","features")) and os.path.exists(os.path.join("Report","labels"))):
        features = joblib.load(os.path.join("Report","features"))
        labels = joblib.load(os.path.join("Report","labels"))
    else:
        features, labels,  = feature_extraction.get_dataset()
    logging.info("Features: {0} - Labels: {1}".format(len(features), len(labels)))
    #clf = classifier.get_classifier(features, labels)
    #joblib.dump(clf,'clf')
    #clf = joblib.load("clf")
    #classifier.test_classifier(clf)
    joblib.dump(features, 'Report\\features')
    joblib.dump(labels, 'Report\\labels')

    image = cv2.imread("test6.jpg")
    draw_image = image.copy()

    logging.info("Fitting an scaler and transforming the features")
    scaler = StandardScaler().fit(features)
    joblib.dump(scaler, os.path.join("Report","scaler"))
    logging.info("Scaler fitted")
    features_scaled = scaler.transform(features)
    logging.info("Features transformed")
    logging.info("Features: {0} - Labels: {1}".format(len(features_scaled), len(labels)))

    logging.info("Searching extra tree classifier params")
    my_clf = best_extratrees(features_scaled, labels, 5, 3)
    logging.info("Best Extra Trees Classifier selected")

    logging.info("Training and making report")
    trained_clf = classifier.KFoldCrossValidation(features_scaled, labels, "Report", my_clf)
    joblib.dump(trained_clf, 'Report\\clf')
    logging.info("Report done!")

    windows = utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    windows.extend( utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(192, 192), xy_overlap=(0.5, 0.5)) )

    windows.extend( utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(144, 144), xy_overlap=(0.5, 0.5)) )

    logging.info("Searching hot windows using classifier")
    hot_windows = utils.search_windows(image, windows, trained_clf, scaler)

    logging.info("Drawing the hot image")
    window_img = utils.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()

    utils.car_positions(image, hot_windows)


if __name__ == '__main__':
    main()