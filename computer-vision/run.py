import pandas as pd
import configparser
from pathlib import Path
import os
from utils import get_data_path, get_cv_config_path, get_output_path, parse_cutoff
from states_classifier import StatesClassifier


def run():
    config = configparser.ConfigParser()
    config_path = get_cv_config_path()
    config.read(str(config_path))

    video_conf = config["video-classification"]
    source = Path(video_conf["filepath"])
    filename = video_conf["filename"]
    video_length = video_conf["video_length"]
    consecutive_frame_count = int(video_conf["consecutive_frame_count"])
    stat_cutoff = parse_cutoff(video_conf["stationary_cutoff"])
    stat_move_cutoff = parse_cutoff(video_conf["stationary_movement_cutoff"])
    render_video = eval(video_conf["render_video"])
    output_video = eval(video_conf["output_video"])
    video_dest = Path(video_conf["output_video_destination"])
    framerate = int(video_conf["framerate"])
    states_dest = get_output_path().joinpath("states").joinpath(source.stem + "_states.csv")

    if not os.path.exists(source):
        source = get_data_path().joinpath("video").joinpath(video_length).joinpath(filename) 
    
    if os.path.exists(video_dest):
        output_dest = Path(video_dest).joinpath(source.name)
    else:
        # Fix output dest None to default or something better
        output_dest = None

    clf = StatesClassifier(stat_cutoff, stat_move_cutoff, fps=framerate)
    states = clf.classify(source, consecutive_frame_count, render_video, output_video, output_dest)
    df = pd.DataFrame(data=states, columns=["state"])
    df.to_csv(states_dest, index=False)


if __name__ == "__main__":
    run()
