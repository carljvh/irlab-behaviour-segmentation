import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_path, get_output_path, read_video, find_max_contour

def background_images():
    data_path = get_data_path().joinpath("backgrounds")
    images = []
    for i in range(4):
        filename = "bg_long_example_%s.jpg" % str(i + 1)
        images.append(cv.imread(str(data_path.joinpath(filename)), cv.IMREAD_COLOR))

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Background images")
    axs = axs.flatten()
    counter = 0
    for img, ax in zip(images, axs):
        counter += 1
        ax.set_title("Video %s" % counter)
        ax.axis("off")
        ax.imshow(img)
    plt.show()


def generate_processing_images():
    bg_path = get_data_path().joinpath("backgrounds")
    background = cv.imread(str(bg_path.joinpath("bg_long_example_4.jpg")), cv.IMREAD_COLOR)
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    source = get_data_path().joinpath("video").joinpath("long").joinpath("long_example_4.mp4")
    cap = read_video(str(source))
    frame_diffs = []
    for i in range(15):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_diff = cv.absdiff(gray, background)
        _, thres = cv.threshold(frame_diff, 100, 255, cv.THRESH_BINARY)
        dilate_frame = cv.dilate(thres, None, iterations=3)
        frame_diffs.append(dilate_frame)


    sum_frames = np.sum(frame_diffs, axis=0, dtype=np.uint8)
    max_contour = find_max_contour(sum_frames)
    blank_img = np.zeros(sum_frames.shape[:2], dtype=np.uint8)
    cv.fillPoly(blank_img, pts=[max_contour], color=(255,255,255))
    dest = get_output_path().joinpath("video").joinpath("frame_dilation.jpg")
    cv.imwrite(str(dest), blank_img)


def calculate_stats():
    states_path = get_output_path().joinpath("all-states")
    cv_states = pd.read_csv(str(states_path.joinpath("cv_states.csv")), header=None).to_numpy()
    kp_states = pd.read_csv(str(states_path.joinpath("kp_states.csv")), header=None).to_numpy()

    absolute = np.abs(cv_states-kp_states)

    ## Percentage of matching total
    zeros = absolute[np.where(absolute == 0)]
    print(zeros.shape[0] / (2400*3))

    counts = []
    for i in range(3):
        count = 0
        for j in range(3):
            cv_indices = list(np.where(cv_states[j,:] == i))
            kp_indices = list(np.where(kp_states[j,:] == i))
            count += sum(cv_indices == kp_indices)
        counts[i] = count
    print(counts)

source = get_output_path().joinpath("video").joinpath("long_example_4.mp4")
cap = read_video(str(source))
cap.set(cv.CAP_PROP_POS_FRAMES, 1390-1)
res, frame = cap.read()
dest = get_output_path().joinpath("images").joinpath("final_video_frame.jpg")
cv.imwrite(str(dest), frame)