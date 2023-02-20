import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from collections import deque
from enum import Enum
from utils import read_video, generate_background, process_frame, get_output_path, find_max_contour


class State(Enum):
    STATIONARY = 0
    STATIONARY_MOVEMENT = 1
    MOVEMENT = 2


class StatesClassifier:
    def __init__(self, stationary_cutoff: int, stationary_movement_cutoff: int, fps: int = 30) -> None:
        self.stationary_cutoff = stationary_cutoff
        self.stationary_movement_cutoff = stationary_movement_cutoff
        self.fps = fps

    def get_output_object(self, cap, dest: Path = None):
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        if dest:
            save_name = str(dest)
        else:
            save_name = str(get_output_path().joinpath("video").joinpath("subject_video.mp4"))
        out = cv.VideoWriter(
            filename=save_name,
            fourcc=cv.VideoWriter_fourcc(*"mp4v"),
            fps=self.fps,
            frameSize=(frame_width, frame_height),
        )
        return out

    def classify_state(self, sum_frames: np.ndarray):
        max_contour = find_max_contour(sum_frames)
        blank_img = np.zeros(sum_frames.shape[:2], dtype=np.uint8)
        cv.fillPoly(blank_img, pts=[max_contour], color=(255,255,255))
        count = cv.countNonZero(blank_img)

        if count < self.stationary_cutoff:
            state = State.STATIONARY
        elif self.stationary_cutoff < count < self.stationary_movement_cutoff:
            state = State.STATIONARY_MOVEMENT
        else:
            state = State.MOVEMENT
        return state

    def render_video_metadata(self, orig_frame: np.ndarray, max_contour, state_name: str):
        (x, y, w, h) = cv.boundingRect(max_contour)
        cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(orig_frame, "Status: {}".format(state_name), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return orig_frame

    def classify(
        self,
        source: Path,
        consecutive_frame_count: int = 5,
        render_video: bool = False,
        output_video: bool = False,
        output_dest: Path = None,
    ) -> np.ndarray:
        cap = read_video(str(source))
        total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        states = np.zeros(total_frame_count, dtype=np.uint8)
        if output_video:
            out = self.get_output_object(cap, output_dest)

        background = generate_background(source)
        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

        frame_diffs = []
        state = State.STATIONARY
        for frame_count in tqdm(range(total_frame_count)):
            ret, frame = cap.read()
            orig_frame = frame.copy()
            if ret == True:
                processed_diff = process_frame(frame, background)
                frame_diffs.append(processed_diff)
                sum_frames = np.sum(frame_diffs, axis=0, dtype=np.uint8)
                max_contour = find_max_contour(sum_frames)

                if (frame_count + 1) % consecutive_frame_count == 0:
                    state = self.classify_state(sum_frames)
                    states[frame_count + 1 - consecutive_frame_count:frame_count] = state.value
                    frame_diffs.clear()

                if render_video:
                    modified_frame = self.render_video_metadata(orig_frame, max_contour, state.name)
                    cv.imshow("subject", modified_frame)
                    if cv.waitKey(10) & 0xFF == ord("q"):
                        break

                if output_video:
                    modified_frame = self.render_video_metadata(orig_frame, max_contour, state.name)
                    out.write(modified_frame)
            else:
                break
        cap.release()
        cv.destroyAllWindows()
        return states
