import numpy as np
import cv2 as cv
import sys
from pathlib import Path
import os
from tqdm import tqdm


def get_project_root():
    return Path(__file__).parent.parent.resolve()


def get_data_path():
    return get_project_root().joinpath("data")


def get_cv_config_path():
    return get_project_root().joinpath("computer-vision").joinpath("config").joinpath("comp_vision.cfg")


def get_output_path():
    return get_project_root().joinpath("output")


def parse_cutoff(cutoff_string: str):
    parts = cutoff_string.split("e")
    cutoff = float(parts[0]) * np.power(10, int(parts[1]))
    return cutoff


def read_video(source: str):
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit()
    else:
        return cap


def process_frame(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_diff = cv.absdiff(gray, background)
    _, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
    dilate_frame = cv.dilate(thres, None, iterations=2)
    return dilate_frame


def find_max_contour(frame_diff_list: list):
    sum_frames = np.sum(frame_diff_list, axis=0, dtype=np.uint8)
    contours, _ = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    tuples = [(contour, cv.contourArea(contour)) for contour in contours]
    max_tuple = max(tuples, key=lambda tup: tup[1])
    max_contour = max_tuple[0]
    return max_contour


def generate_background(source: Path, dest: Path = None, frame_sample_count: int = 120) -> np.ndarray:
    background_source = get_data_path().joinpath("backgrounds")
    background_path = background_source.joinpath("bg_" + str(source.stem) + ".jpg")

    if os.path.exists(background_path):
        return cv.imread(str(background_path), cv.IMREAD_COLOR)
    else:
        cap = cv.VideoCapture(str(source))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_indices = np.random.choice(frame_count, frame_sample_count)
        frames = []
        for idx in tqdm(frame_indices):
            cap.set(cv.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            frames.append(frame)

        median_frame = np.median(frames, axis=0).astype(np.uint8)
        cap.release()
        if dest:
            cv.imwrite(str(dest), median_frame)
        else:
            cv.imwrite(background_source, median_frame)
        return median_frame
