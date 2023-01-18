import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import os
from tqdm import tqdm
from collections import deque
import configparser
from utils import (
    get_data_path,
    get_cv_config_path,
    get_output_path,
    read_video,
    generate_background,
    process_frame,
    find_max_contour,
)


def count_white_pixels(frame_diffs: list):
    sum_frames = np.sum(frame_diffs, axis=0, dtype=np.uint8)
    max_contour = find_max_contour(sum_frames)
    blank_img = np.zeros(sum_frames.shape[:2], dtype=np.uint8)
    cv.fillPoly(blank_img, pts=[max_contour], color=(255,255,255))
    count = cv.countNonZero(blank_img)
    return count


def get_changed_pixel_counts(source_folder: Path, consecutive_frame_count: int = 5) -> list:
    total_counts = []

    for filename in os.listdir(source_folder):
        source = source_folder.joinpath(filename)
        background = generate_background(source)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        cap = read_video(str(source))
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        delta_size = int(np.ceil(total_frame_count/consecutive_frame_count))
        delta_counts = np.zeros(delta_size, dtype=np.uint64)

        frame_diffs = []
        for frame_count in tqdm(range(total_frame_count)):
            ret, frame = cap.read()
            if ret == True:
                processed_diff = process_frame(frame, background)
                frame_diffs.append(processed_diff)
                if (frame_count + 1) % consecutive_frame_count == 0:
                    count_idx = int((frame_count+1) / consecutive_frame_count - 1)
                    delta_counts[count_idx] = count_white_pixels(frame_diffs)
                    frame_diffs.clear()
            else:
                delta_counts[-1] = count_white_pixels(frame_diffs)
                break
        total_counts.extend(delta_counts.tolist())
        cap.release()

    return total_counts


def main():
    config = configparser.ConfigParser()
    config_path = get_cv_config_path()
    config.read(str(config_path))

    distr_conf = config["distribution-generation"]
    source_folder = Path(distr_conf["source_folder"])
    video_length = distr_conf["video_length"]
    n_bins = int(distr_conf["number_of_bins"])

    if not os.path.exists(source_folder):
        source_folder = get_data_path().joinpath("video").joinpath(video_length)
    output_path = get_output_path().joinpath("histograms")
    consecutive_frame_count = int(config["video-classification"]["consecutive_frame_count"])

    total_counts = get_changed_pixel_counts(source_folder, consecutive_frame_count)
    sns.histplot(data=total_counts, bins=n_bins)
    plt.savefig(output_path.joinpath("pixel_deltas_%s_%s.png" % (video_length, consecutive_frame_count)), format="png")


if __name__ == "__main__":
    main()
