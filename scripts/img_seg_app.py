import json
import torch

from scripts.img_seg_model import NeuralNetwork, ModelOps


class ImageSegApplication:
    """Class loading settings from config file and starting either training or inference"""

    config_path = "scripts/config.json"
    def __init__(self):
        self._load_json_config(ImageSegApplication.config_path)

    def _load_json_config(self, path):
        with open(path, "r") as file:
            temp = json.load(file)
        print(temp)
        ModelOps.batch_size = temp["Training_settings"]["batch_size"]
        ModelOps.number_of_epochs = temp["Training_settings"]["number_of_epochs"]
        ModelOps.model_path = temp["Training_settings"]["model_path"]
        ModelOps.number_of_workers = temp["Training_settings"]["number_of_workers"]
        ModelOps.train_data_size =  temp["Training_settings"]["train_data_size"]

        ModelOps.font_file = temp["Training_settings"]["font_file"]
        ModelOps.dataset_name = temp["Training_settings"]["dataset_name"]
        ModelOps.dataset_dir = temp["Training_settings"]["dataset_dir"]
        ModelOps.images_subdirectory = temp["Training_settings"]["images_subdirectory"]
        ModelOps.annotation_file_path = temp["Training_settings"]["annotation_file_path"]
        ModelOps.project_name = temp["Training_settings"]["project_name"]
        ModelOps.train_pct = temp["Training_settings"]["training_split_ratio"]
        ModelOps.initial_learning_rate = temp["Training_settings"]["initial_learning_rate"]

        ModelOps.prediction_threshold = temp["Prediction_settings"]["threshold"]

    def show_sample_data(self):
        ModelOps.load_sample_image()

    def run_training(self):

        # load data
        ModelOps.load_data()

        # run training and testing
        ModelOps.run_epochs()

