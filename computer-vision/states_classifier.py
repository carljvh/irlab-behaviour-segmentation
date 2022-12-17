import numpy as np
import cv2 as cv
from pathlib import Path
import os
from tqdm import tqdm
import sys
from collections import deque

class StatesClassifier:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    @staticmethod
    def parse_cutoff(cutoff_string:str):
        parts = cutoff_string.split("e")
        cutoff = float(parts[0]) * np.power(10, int(parts[1]))
        return cutoff

    @staticmethod
    def read(source: str):
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            print("Error opening video stream or file")
            sys.exit()
        else:
            return cap

    @staticmethod
    def process(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_diff = cv.absdiff(gray, background)
        _, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
        dilate_frame = cv.dilate(thres, None, iterations=2)
        return dilate_frame

    @staticmethod
    def get_background(source: Path, dest: Path = None, frame_sample_count: int = 120, write=True) -> np.ndarray:
        if os.path.exists(dest):
            return cv.imread(str(dest), cv.IMREAD_COLOR)
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
            if write:
                cv.imwrite(str(dest), median_frame)
            return median_frame

    def detect(self, source: Path, dest: Path, stationary_cutoff, stationary_moving_cutoff, consecutive_frame_count: int = 5) -> None:
        cap = self.read(str(source))
        """
        # define codec and create VideoWriter object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        save_name = str(dest.joinpath("output_test"))
        #out = cv.VideoWriter(save_name, cv.VideoWriter_fourcc(*"mp4v"), 10, (frame_width, frame_height))
        """
        dest = self.data_path.joinpath("backgrounds").joinpath("bg_" + str(source.stem) + ".jpg")
        background = self.get_background(source, dest)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        frame_diff_list = deque(maxlen=5)
        for _ in range(consecutive_frame_count):
            _, frame = cap.read()
            processed_diff = self.process(frame, background)
            frame_diff_list.append(processed_diff)
        
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        for frame_count in tqdm(range(total_frame_count-consecutive_frame_count)):
            _, frame = cap.read()
            orig_frame = frame.copy()

            frame_diff_list.popleft()
            processed_diff = self.process(frame, background)
            frame_diff_list.append(processed_diff)

            sum_frames = sum(list(frame_diff_list))
            contours, _ = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            tuples = [(contour, cv.contourArea(contour)) for contour in contours]
            max_tuple = max(tuples, key=lambda tup: tup[1])
            max_contour = max_tuple[0]
            (x, y, w, h) = cv.boundingRect(max_contour)
            cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            mean_frames = np.mean(list(frame_diff_list), axis=0).astype(np.uint8)
            mean_diff_count = np.sum(mean_frames)
            if mean_diff_count < stationary_cutoff:
                state = "STATIONARY"
            elif stationary_cutoff < mean_diff_count < stationary_moving_cutoff:
                state = "STATIONARY MOVEMENT"
            else:
                state = "MOVEMENT"
                
            cv.putText(orig_frame, "Status: {}".format(state), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)
            cv.imshow("subject", orig_frame)
            if cv.waitKey(100) & 0xFF == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()