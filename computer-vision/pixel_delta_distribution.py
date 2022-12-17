import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import os
from tqdm import tqdm
from states_classifier import StatesClassifier as clf

DATA_PATH = Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT_PATH = Path(__file__).parent.parent.resolve().joinpath("output")


def get_changed_pixel_counts(source_folder: Path, consecutive_frame_count: int = 5) -> list:
    total_counts = []

    for filename in os.listdir(source_folder):
        source = source_folder.joinpath(filename)
        dest = DATA_PATH.joinpath("backgrounds").joinpath("bg_" + str(source.stem) + ".jpg")
        
        background = clf.get_background(source, dest)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        cap = clf.read(str(source))
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        mean_diff_counts = np.zeros(total_frame_count-consecutive_frame_count+1, dtype=float)

        frame_diff_list = []
        for _ in range(consecutive_frame_count):
            _, frame = cap.read()
            processed_diff = clf.process(frame, background)
            frame_diff_list.append(processed_diff)
        mean_diff = np.mean(frame_diff_list, axis=0).astype(np.uint8)
        mean_diff_counts[0] = np.sum(mean_diff)

        for frame_count in tqdm(range(total_frame_count-consecutive_frame_count)):
            _, frame = cap.read()
            frame_diff_list.pop(0)
            processed_diff = clf.process(frame, background)
            frame_diff_list.append(processed_diff)
            mean_diff = np.mean(frame_diff_list, axis=0).astype(np.uint8)
            mean_diff_counts[frame_count+1] = np.sum(mean_diff)
        total_counts.extend(mean_diff_counts.tolist())
        cap.release()

    return total_counts


def main():
    total_counts = get_changed_pixel_counts(DATA_PATH.joinpath("video").joinpath("short"))
    sns.histplot(data=total_counts, bins=100)
    plt.savefig(OUTPUT_PATH.joinpath("pixel_deltas.png"), format="png")


if __name__ == "__main__":
    main()