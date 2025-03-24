import json
import torch
from dataclasses import dataclass

from scripts.img_seg_model import NeuralNetwork, ModelOps

@dataclass
class TrainingSettings:
    batch_size: int = 0
    number_of_epochs: int = 0

    train_data_path: str = ""
    test_data_path: str = ""

    model_path: str = ""

class ImageSegApplication:
    config_path = "scripts/config.json"
    def __init__(self):
        self.training_settings = TrainingSettings()
        self._load_json_config(ImageSegApplication.config_path)


    def _load_json_config(self, path):
        with open(path, "r") as file:
            temp = json.load(file)
        print(temp)
        self.training_settings.batch_size = temp["Training_settings"]["batch_size"]
        self.training_settings.number_of_epochs = temp["Training_settings"]["number_of_epochs"]
        self.training_settings.train_data_path = temp["Training_settings"]["train_data_path"]
        self.training_settings.test_data_path = temp["Training_settings"]["test_data_path"]
        self.training_settings.model_path = temp["Training_settings"]["model_path"]

    def show_sample_data(self):
        ModelOps.load_sample_image()

    def run_training(self):

        # select device
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {device} device")

        # define model
        model = NeuralNetwork().to(device)
        print(model)

        # configure training and testing
        ModelOps.batch_size = self.training_settings.batch_size
        ModelOps.number_of_epochs = self.training_settings.number_of_epochs

        ModelOps.training_data_path = self.training_settings.train_data_path
        ModelOps.testing_data_path = self.training_settings.test_data_path

        # load data
        ModelOps.load_data()

        # run training and testing
        ModelOps.run_epochs(model, device)

        # save model
        model_path = self.training_settings.model_path
        torch.save(model.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}")
