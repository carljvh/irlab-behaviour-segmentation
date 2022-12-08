import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import os
from states_classifier import StatesClassifier as clf

DATA_PATH = Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT_PATH = Path(__file__).parent.parent.resolve().joinpath("output")


def get_changed_pixel_counts(source_folder: Path, consecutive_frame_count: int = 5) -> list:
    total_counts = []

    for filename in os.listdir(source_folder):
        source = source_folder.joinpath(filename)
        cap = clf.read(str(source))
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        mean_diff_counts = np.zeros(int(np.floor(total_frame_count / consecutive_frame_count)), dtype=float)

        dest = DATA_PATH.joinpath("backgrounds").joinpath(str(source.stem) + "_background.jpg")
        background = clf.get_background(source, dest)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_count += 1

                if frame_count % consecutive_frame_count == 0 or frame_count == 1:
                    frame_diff_list = []
                processed_diff = clf.process(frame, background)
                frame_diff_list.append(processed_diff)

                if len(frame_diff_list) == consecutive_frame_count:
                    mean_diff = np.mean(frame_diff_list, axis=0).astype(np.uint8)
                    count_idx = int(frame_count/consecutive_frame_count)-1
                    mean_diff_counts[count_idx] = np.sum(mean_diff)

                if frame_count % 100 == 0:
                    print("iteration: %d / %d" % (frame_count, total_frame_count))
            else:
                break

        cap.release()
        total_counts.extend(mean_diff_counts.tolist())

    return total_counts


def main():
    total_counts = get_changed_pixel_counts(DATA_PATH.joinpath("video"))
    sns.histplot(data=total_counts, bins=100)
    plt.savefig(OUTPUT_PATH.joinpath("frequency-results").joinpath("delta_histogram.png"), format="png")
    plt.show()


if __name__ == "__main__":
    main()