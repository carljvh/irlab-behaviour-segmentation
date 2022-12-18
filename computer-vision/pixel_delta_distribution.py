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
)


def get_changed_pixel_counts(source_folder: Path, consecutive_frame_count: int = 5) -> list:
    total_counts = []

    for filename in os.listdir(source_folder):
        source = source_folder.joinpath(filename)
        background = generate_background(source)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        cap = read_video(str(source))
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        mean_diff_counts = np.zeros(total_frame_count - consecutive_frame_count + 1, dtype=float)

        frame_diffs = deque(maxlen=consecutive_frame_count)
        for _ in range(consecutive_frame_count):
            _, frame = cap.read()
            processed_diff = process_frame(frame, background)
            frame_diffs.append(processed_diff)
        mean_diff = np.mean(list(frame_diffs), axis=0).astype(np.uint8)
        mean_diff_counts[0] = np.sum(mean_diff)

        for frame_count in tqdm(range(total_frame_count - consecutive_frame_count)):
            _, frame = cap.read()
            frame_diffs.popleft()
            processed_diff = process_frame(frame, background)
            frame_diffs.append(processed_diff)
            mean_diff = np.mean(list(frame_diffs), axis=0).astype(np.uint8)
            mean_diff_counts[frame_count + 1] = np.sum(mean_diff)
        total_counts.extend(mean_diff_counts.tolist())
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
    plt.savefig(output_path.joinpath("pixel_deltas_%s.png" % video_length), format="png")


if __name__ == "__main__":
    main()
