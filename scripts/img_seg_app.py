import json
import math
import os.path
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.img_seg_Ops import ModelOps
from scripts.project_utils import prepare_learning_graph, OrientationCorrection

_DEBUG = False

class ImageSegApplication:
    """Class loading settings from config file and starting either training or inference"""

    default_config_path = "scripts/config.json"

    def __init__(self, config: str = None):
        self.model_ops = None
        if not config:
            self.config_path = ImageSegApplication.default_config_path
        else:
            self.config_path = config

    def _load_json_config(self, path, infer=False):
        """Loading config file with settings from path specified in cls field [config_path]"""
        with open(path, "r") as file:
            temp = json.load(file)

        self.model_ops.state["batch_size"] = temp["Training_settings"]["batch_size"]
        self.model_ops.state["number_of_epochs"] = temp["Training_settings"]["number_of_epochs"]
        self.model_ops.state["model_path"] = temp["Training_settings"]["model_path"]
        self.model_ops.state["number_of_workers"] = temp["Training_settings"]["number_of_workers"]
        self.model_ops.state["train_data_size"] =  temp["Training_settings"]["train_data_size"]

        self.model_ops.state["project_name"] = temp["Training_settings"]["project_name"]
        self.model_ops.state["train_pct"] = temp["Training_settings"]["training_split_ratio"]
        self.model_ops.state["initial_learning_rate"] = temp["Training_settings"]["initial_learning_rate"]

        self.model_ops.state["dataset_name"] = temp["Dataset"]["dataset_name"]
        self.model_ops.state["dataset_dir"] = temp["Dataset"]["dataset_dir"]
        self.model_ops.state["images_subdirectory"] = temp["Dataset"]["images_subdirectory"]
        self.model_ops.state["annotation_file_path"] = temp["Dataset"]["annotation_file_path"]

        self.model_ops.state["prediction_threshold"] = temp["Prediction_settings"]["threshold"]

        self.model_ops.state["font_file"] = temp["Miscellaneous"]["font_file"]

        self.model_ops.state["prediction_model_path"] = temp["Prediction_settings"]["prediction_model_path"]
        self.model_ops.state["image_path"] = temp["Prediction_settings"]["image_path"]
        self.model_ops.state["class_names"] = temp["Training_settings"]["class_names"]

        orientation_corr = OrientationCorrection()
        orientation_corr.define_height_width(
            height=temp["Training_settings"]["image_height"],
            width=temp["Training_settings"]["image_width"],
        )
        self.model_ops.state["orientation_corr"] = orientation_corr

        self.model_ops.state["model_name"] = temp["Model"]["model_name"]

        if infer:
            self.model_ops.state["pretrained"] = True
        else:
            self.model_ops.state["pretrained"] = temp["Training_settings"]["pretrained"]

    def show_sample_data(self):
        self.model_ops.load_sample_image()

    def run_epochs(self, queue: Queue):
        self.model_ops = ModelOps()
        self._load_json_config(self.config_path)

        self.model_ops.load_data()
        self.model_ops.queue = queue
        self.model_ops.get_ready()

        if self.model_ops.state["pretrained"]:
            dir_path = Path(self.model_ops.state["prediction_model_path"]).parent / "log.csv"
            df = pd.read_csv(dir_path)
            temp_path = Path(self.model_ops.state["checkpoint_path"].parent / 'log.csv')
            df.to_csv(temp_path, index=False, header=True)
            print(f"saved to {temp_path}")
            queue.put(temp_path)

        self.model_ops.run_epochs()

    def run_training(self):

        queue = Queue()
        # run training and testing
        p_model_training = Process(target=self.run_epochs, args=(queue,))
        p_display_progress = Process(target=self.display_progress, args=(queue,))

        p_model_training.start()
        p_display_progress.start()

        p_model_training.join()

        queue.put("Kill")

        p_display_progress.join()


    def run_inference(self):
        self.model_ops = ModelOps()
        self._load_json_config(self.config_path, infer=True)
        self.model_ops.infer()

    def _dummy_metadata_process(self, queue: Queue):

        if os.path.exists("log.csv"):
            os.remove("log.csv")

        columns = ['epoch', 'train_loss', 'valid_loss', 'learning_rate', 'model_architecture']
        epoch_df = pd.DataFrame({}, columns=columns)
        temp_path = Path('log.csv')
        epoch_df.to_csv(temp_path, mode='a', index=False, header=True)

        for i in range(5):
            epoch_metadata = {
                'epoch': [i],
                'train_loss': [math.cos(i)],
                'valid_loss': [math.cos(i)],
            }
            epoch_df = pd.DataFrame(epoch_metadata)
            temp_path = Path('log.csv')
            epoch_df.to_csv(temp_path, mode='a', index=False, header=False)

            queue.put(temp_path)
            print("sent data")
            time.sleep(1)

    def test_sample(self):
        self.model_ops.test_on_random_sample()

    def display_progress(self, queue: Queue):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        while True:
            try:
                if len(plt.get_fignums()) == 0:
                    fig, ax = plt.subplots()

                path = queue.get(timeout=0.1)

                if path == "Kill":
                    break
                if _DEBUG:
                    print(f"received: {path}")

                all_data = pd.read_csv(path, index_col=0)
                training_data = all_data[["train_loss", "valid_loss"]]

                ax.clear()  # Clear previous data
                ax.plot(training_data)
                plt.draw()  # Draw the updated plot
                plt.pause(0.1)  # Allow GUI event loop to update
            except Empty:
                pass
            plt.gcf().canvas.flush_events()

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the final plot displayed


if __name__ == "__main__":
    app = ImageSegApplication("config.json")
    queue = Queue()
    # run training and testing

    p_model_training = Process(target=app._dummy_metadata_process, args=(queue,))
    p_display_progress = Process(target=app.display_progress, args=(queue,))

    p_model_training.start()
    p_display_progress.start()

    p_model_training.join()

    queue.put("Kill")

    p_display_progress.join()
