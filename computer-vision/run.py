from pathlib import Path
import configparser
from states_classifier import StatesClassifier

CONFIG_PATH = Path(__file__).parent.resolve().joinpath("config").joinpath("comp_vision.cfg")
DATA_PATH = Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT_PATH = Path(__file__).parent.parent.resolve().joinpath("output")


def run():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    clf = StatesClassifier(DATA_PATH)
    stationary_cutoff = clf.parse_cutoff(config["cutoff-frequencies"]["stationary_cutoff"])
    stationary_moving_cutoff = clf.parse_cutoff(config["cutoff-frequencies"]["stationary_moving_cutoff"])
    clf.detect(DATA_PATH.joinpath("video").joinpath("example_2.mp4"), OUTPUT_PATH, stationary_cutoff, stationary_moving_cutoff)


if __name__ == "__main__":
    run()
