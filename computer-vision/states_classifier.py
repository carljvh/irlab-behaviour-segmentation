import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from collections import deque
from enum import Enum
from utils import read_video, generate_background, process_frame, get_output_path


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

    def render_video_metadata(self, frame_diff_list, orig_frame, state_name):
        sum_frames = sum(frame_diff_list)
        contours, _ = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        tuples = [(contour, cv.contourArea(contour)) for contour in contours]
        max_tuple = max(tuples, key=lambda tup: tup[1])
        max_contour = max_tuple[0]
        (x, y, w, h) = cv.boundingRect(max_contour)
        cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(
            orig_frame, "Status: {}".format(state_name), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3
        )
        return orig_frame

    def classify_state(self, frame_diff_list: list):
        mean_frames = np.mean(frame_diff_list, axis=0).astype(np.uint8)
        mean_diff_count = np.sum(mean_frames)
        if mean_diff_count < self.stationary_cutoff:
            state = State.STATIONARY
        elif self.stationary_cutoff < mean_diff_count < self.stationary_movement_cutoff:
            state = State.STATIONARY_MOVEMENT
        else:
            state = State.MOVEMENT
        return state

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

        for frame_count in tqdm(range(total_frame_count - consecutive_frame_count)):
            _, frame = cap.read()
            orig_frame = frame.copy()

            frame_diffs.popleft()
            processed_diff = process_frame(frame, background)
            frame_diffs.append(processed_diff)

            state = self.classify_state(list(frame_diffs))
            states[frame_count + consecutive_frame_count] = state.value

            if render_video:
                modified_frame = self.render_video_metadata(list(frame_diffs), orig_frame, state.name)
                cv.imshow("subject", modified_frame)
                if cv.waitKey(10) & 0xFF == ord("q"):
                    break

            if output_video:
                modified_frame = self.render_video_metadata(list(frame_diffs), orig_frame, state.name)
                out.write(modified_frame)

        cap.release()
        cv.destroyAllWindows()
        return states
