import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import logging
import os

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    Input image should be gray
    """

    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=False,
                      visualise=vis, feature_vector=feature_vec)

    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features

def get_features(img_filename):
    img = cv2.imread(img_filename)
    features = get_features_from_img(img)
    return features

def get_features_from_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    file_features = []
    hog_features, _ = get_hog_features(gray, orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     vis=True, feature_vec=False)
    spatial_features = bin_spatial(img)
    hist_features = color_hist(img)

    file_features.append(hog_features.ravel())
    file_features.append(spatial_features)
    file_features.append(hist_features)

    features = np.concatenate(file_features)
    return features

def get_small_dataset():
    features = []
    labels = []
    car_images = glob.glob(os.path.join('vehicles_smallset','cars1*.jpeg'))
    car_images.extend(glob.glob(os.path.join('vehicles_smallset','cars2','*.jpeg')))

    noncar_images = glob.glob(os.path.join('non-vehicles_smallset','notcars1','*.jpeg'))
    noncar_images.extend(glob.glob(os.path.join('non-vehicles_smallset','notcars2','*.jpeg')))

    for car in car_images:
        labels.append("car")
        features.append(get_features(car))

    for noncar in noncar_images:
        labels.append("noncar")
        features.append(get_features(noncar))

    logging.info("Array of features contains {0} elements".format(len(features)))
    logging.info("Array of labels contains {0} elements".format(len(labels)))
    return features, labels

def get_dataset():
    features = []
    labels = []
    car_images = glob.glob(os.path.join('vehicles','*','*.png'))

    noncar_images = glob.glob(os.path.join('non-vehicles','*','*.png'))

    for car in car_images:
        labels.append("car")
        features.append(get_features(car))

    for noncar in noncar_images:
        labels.append("noncar")
        features.append(get_features(noncar))

    logging.info("Array of features contains {0} elements".format(len(features)))
    logging.info("Array of labels contains {0} elements".format(len(labels)))
    return features, labels

def get_test():
    features = []
    labels = []
    car_images = glob.glob(os.path.join('vehicles_smallset','cars3','*.jpeg'))
    noncar_images = glob.glob(os.path.join('non-vehicles_smallset','notcars3','*.jpeg'))

    for car in car_images:
        labels.append("car")
        features.append(get_features(car))

    for noncar in noncar_images:
        labels.append("noncar")
        features.append(get_features(noncar))

    logging.info("Array of features contains {0} elements".format(len(features)))
    logging.info("Array of labels contains {0} elements".format(len(labels)))
    return features, labels