import cv2
import numpy as np
import utils
import classifier
import feature_extraction
import matplotlib.pyplot as plt
import joblib


#Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

#Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

#Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

#Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

#Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
## and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

#Estimate a bounding box for vehicles detected.

def main():
    features, labels = feature_extraction.get_dataset()
    clf = classifier.get_classifier(features, labels)
    joblib.dump(clf,'clf')
    #clf = joblib.load("clf")
    classifier.test_classifier(clf)

    image = cv2.imread("test6.jpg")
    draw_image = image.copy()


    windows = utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    windows.extend( utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(192, 192), xy_overlap=(0.5, 0.5)) )

    windows.extend( utils.slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
                        xy_window=(144, 144), xy_overlap=(0.5, 0.5)) )

    hot_windows = utils.search_windows(image, windows, clf)

    window_img = utils.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()


    utils.car_positions(image, hot_windows)


if __name__ == '__main__':
    main()