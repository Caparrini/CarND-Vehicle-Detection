import logging
import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from utils import slide_window, search_windows, draw_boxes, car_positions
import os
import joblib
import glob
logging.basicConfig(level=logging.DEBUG)


from moviepy.editor import VideoFileClip

trained_clf = joblib.load(os.path.join("Report","clf"))
scaler = joblib.load(os.path.join("Report","scaler"))
#trained_clf = joblib.load('clf')
#scaler = joblib.load('scaler')

def process_frame(image):
    global trained_clf
    global scaler

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    draw_image = image.copy()

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[360, 700],
                        xy_window=(64, 64), xy_overlap=(0.85, 0.85))

    logging.info("Searching hot windows using classifier")
    hot_windows = search_windows(image, windows, trained_clf, scaler)

    logging.info("Drawing the hot image")
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    #plt.imshow(window_img)
    #plt.show()

    return car_positions(image, hot_windows)

def test_images(folder="../test_images"):
    images_files = glob.glob(os.path.join(folder, "test*.jpg"))
    images = [cv2.cvtColor(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB),cv2.COLOR_BGR2RGB) for x in images_files]
    #images_undistorted = [cal_cam.undistort(x) for x in images_files]
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 3
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        img = images.pop()
        img = process_frame(img)
        plt.imshow(img)
    plt.show()

def main():
    project_video_input = os.path.join("..","project_video.mp4")
    project_video_output = os.path.join("..","output_video.mp4")

    clip = VideoFileClip(project_video_input)
    project_clip = clip.fl_image(process_frame)
    project_clip.write_videofile(project_video_output, audio=False)


if __name__ == '__main__':
    main()