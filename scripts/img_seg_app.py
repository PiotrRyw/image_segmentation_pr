import json
import torch

from scripts.img_seg_model import ModelOps


class ImageSegApplication:
    """Class loading settings from config file and starting either training or inference"""

    config_path = "scripts/config.json"
    def __init__(self):
        self._load_json_config(ImageSegApplication.config_path)

    def _load_json_config(self, path):
        """Loading config file with settings from path specified in cls field [config_path]"""
        with open(path, "r") as file:
            temp = json.load(file)
        print(temp)
        ModelOps.state["batch_size"] = temp["Training_settings"]["batch_size"]
        ModelOps.state["number_of_epochs"] = temp["Training_settings"]["number_of_epochs"]
        ModelOps.state["model_path"] = temp["Training_settings"]["model_path"]
        ModelOps.state["number_of_workers"] = temp["Training_settings"]["number_of_workers"]
        ModelOps.state["train_data_size"] =  temp["Training_settings"]["train_data_size"]

        ModelOps.state["project_name"] = temp["Training_settings"]["project_name"]
        ModelOps.state["train_pct"] = temp["Training_settings"]["training_split_ratio"]
        ModelOps.state["initial_learning_rate"] = temp["Training_settings"]["initial_learning_rate"]

        ModelOps.state["dataset_name"] = temp["Dataset"]["dataset_name"]
        ModelOps.state["dataset_dir"] = temp["Dataset"]["dataset_dir"]
        ModelOps.state["images_subdirectory"] = temp["Dataset"]["images_subdirectory"]
        ModelOps.state["annotation_file_path"] = temp["Dataset"]["annotation_file_path"]

        ModelOps.state["prediction_threshold"] = temp["Prediction_settings"]["threshold"]

        ModelOps.state["font_file"] = temp["Miscellaneous"]["font_file"]

        ModelOps.state["prediction_model_path"] = temp["Prediction_settings"]["prediction_model_path"]
        ModelOps.state["image_path"] = temp["Prediction_settings"]["image_path"]
        ModelOps.state["class_names"] = temp["Training_settings"]["class_names"]

    def show_sample_data(self):
        ModelOps.load_sample_image()

    def run_training(self):

        # load data
        ModelOps.load_data()

        # run training and testing
        ModelOps.run_epochs()

    def run_inference(self):
        ModelOps.infer()

    def test_sample(self):
        ModelOps.test_on_random_sample()
