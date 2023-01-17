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
        self.mean_diff_count = 0

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

    def classify_state(self, frame_diff_list: list):
        mean_diff = np.mean(frame_diff_list, axis=0).astype(np.uint8)
        max_contour = find_max_contour(frame_diff_list)
        cv.drawContours(mean_diff, max_contour, -1, 0, -1)
        new_diff_count = cv.countNonZero(mean_diff)
        delta_count = np.abs(new_diff_count - self.mean_diff_count)
        self.mean_diff_count = new_diff_count

        if delta_count < self.stationary_cutoff:
            state = State.STATIONARY
        elif self.stationary_cutoff < delta_count < self.stationary_movement_cutoff:
            state = State.STATIONARY_MOVEMENT
        else:
            state = State.MOVEMENT
        return state

    def render_video_metadata(self, orig_frame: np.ndarray, frame_diff_list: list, state_name: str):
        max_contour = find_max_contour(frame_diff_list)
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

        frame_diffs = deque(maxlen=consecutive_frame_count)
        for _ in range(consecutive_frame_count):
            _, frame = cap.read()
            processed_diff = process_frame(frame, background)
            frame_diffs.append(processed_diff)
        state = self.classify_state(list(frame_diffs))
        states[: consecutive_frame_count - 1] = state.value

        for frame_count in tqdm(range(consecutive_frame_count, total_frame_count)):
            _, frame = cap.read()
            orig_frame = frame.copy()

            frame_diffs.popleft()
            processed_diff = process_frame(frame, background)
            frame_diffs.append(processed_diff)

            state = self.classify_state(list(frame_diffs))
            states[frame_count] = state.value

            if render_video:
                modified_frame = self.render_video_metadata(orig_frame, list(frame_diffs), state.name)
                cv.imshow("subject", modified_frame)
                if cv.waitKey(10) & 0xFF == ord("q"):
                    break

            if output_video:
                modified_frame = self.render_video_metadata(orig_frame, list(frame_diffs), state.name)
                out.write(modified_frame)

        cap.release()
        cv.destroyAllWindows()
        return states
