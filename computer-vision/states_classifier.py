import numpy as np
import cv2 as cv
from pathlib import Path
import os
from tqdm import tqdm
import sys

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
        ret, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
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
                ret, frame = cap.read()
                frames.append(frame)

            median_frame = np.median(frames, axis=0).astype(np.uint8)
            cap.release()
            if write:
                cv.imwrite(str(dest), median_frame)
            return median_frame

    def detect(self, source: Path, dest: Path, stationary_cutoff, stationary_moving_cutoff, consecutive_frame_count: int = 3) -> None:
        cap = self.read(str(source))
        """
        # define codec and create VideoWriter object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        save_name = str(dest.joinpath("output_test"))
        #out = cv.VideoWriter(save_name, cv.VideoWriter_fourcc(*"mp4v"), 10, (frame_width, frame_height))
        """

        # background = self.get_aggregated_background(source.parent)
        dest = self.data_path.joinpath("backgrounds").joinpath(str(source.stem) + "_background.jpg")
        background = self.get_background(source, dest)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_count += 1
                orig_frame = frame.copy()

                if frame_count % consecutive_frame_count == 0 or frame_count == 1:
                    frame_diff_list = []
                processed_diff = self.process(frame, background)
                frame_diff_list.append(processed_diff)

                if len(frame_diff_list) == consecutive_frame_count:
                    sum_frames = sum(frame_diff_list)
                    mean_frames = np.mean(frame_diff_list, axis=0).astype(np.uint8)
                    mean_diff_count = np.sum(mean_frames)
                    
                    # TODO: Replace contours with highest, lowest, leftmost, rightmost pixel change?
                    contours, _ = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    tuples = [(contour, cv.contourArea(contour)) for contour in contours]
                    max_tuple = max(tuples, key=lambda tup: tup[1])
                    max_contour = max_tuple[0]

                    (x, y, w, h) = cv.boundingRect(max_contour)
                    cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            else:
                break
        cap.release()
        cv.destroyAllWindows()
