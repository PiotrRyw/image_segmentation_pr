import json
import random
import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from cjm_pandas_utils.core import markdown_to_pandas
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_bounding_boxes
from torchtnt.utils import get_module_summary

from cjm_psl_utils.core import download_file

from functools import partial
from distinctipy import distinctipy

from scripts.img_seg_dataset import COCOLIVECellDataset, DatasetUtils
from scripts.project_utils import train_tfms, verify_dataset, tuple_batch, draw_learning_graph
from scripts.segmenting_images import segment_image
from scripts.test_model import test_model
from scripts.train import train_loop
from scripts.network_design import NeuralNetworkOpsBaseClass, NeuralNetworkOps


class State:
    """class holding state of the app. It is instanced with a dict with keys and all their values equal to None
    Each key's value can ONLY be SET ONCE
    """
    def __init__(self):
        """State object holds values of properties in _state_dict [dict] alongside a _is_set [dict] which keeps track
        whether a property has been set since the creation of State object.
        """
        self._state_dict = {
            "batch_size": 0,
            "number_of_epochs": 0,
            "number_of_workers": 0,
            "train_data_size": 0,
            "training_data": Dataset,
            "test_data": Dataset,
            "dataloader_training_data": DataLoader,
            "dataloader_testing_data": DataLoader,
            "font_file": "",
            "colors": [],
            "initial_learning_rate": 0.0,
            "dataset_name": "",
            "dataset_dir": "",
            "images_subdirectory": "",
            "annotation_file_path": "",
            "project_dir": Path(),
            "model_path": "",
            "project_name": "",
            "train_pct": 0.0,
            "class_names": [],
            "dataset_path": Path(),
            "validation_keys": [],
            "annotation_df": pd.DataFrame(),
            "model": torchvision.models,
            "device": NeuralNetworkOpsBaseClass(),
            "image_dict": {},
            "prediction_threshold": 0.0,
            "checkpoint_dir": Path(),
            "prediction_model_path": "",
            "image_path": ""
        }
        keys = self._state_dict.keys()
        self._is_set = dict(zip(keys, [False for _ in range(len(keys))]))

    def __getitem__(self, item):
        if self._is_set[item] is False:
            raise RuntimeError(f"property {item} has not been set yet")
        return self._state_dict[item]

    def __setitem__(self, item, data):
        if not self._is_set[item]:
            self._state_dict[item] = data
            self._is_set[item] = True
        else:
            msg = f"property {item} cannot be set as it already has a value of {self._state_dict[item]}"
            raise RuntimeError(msg)

    def check_properties(self, required):
        is_good_to_go = True
        missing_attributes = []
        for req in required:
            if not self._is_set[req]:
                missing_attributes.append(req)
        return is_good_to_go, missing_attributes

    @property
    def data_can_be_loaded(self) -> Tuple[bool, List]:
        required = ["batch_size", "number_of_epochs", "dataset_name", "dataset_dir", "images_subdirectory",
                    "annotation_file_path", "project_name", "train_pct"
                    ]

        return self.check_properties(required)

    @property
    def data_is_loaded(self) -> Tuple[bool, List]:
        required = ["training_data", "test_data"]

        return self.check_properties(required)


class ModelOpsUtils:
    @staticmethod
    def make_config_dir_save_color_map(color_map, checkpoint_dir, model_name, dataset_path_name):
        # Create the checkpoint directory if it does not already exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(f"{checkpoint_dir}/{dataset_path_name}-colormap.json", "w") as file:
            json.dump(color_map, file)

        # Print the name of the file that the color map was written to
        print(f"{checkpoint_dir}/{dataset_path_name}-colormap.json")

class ModelOps:
    state = State()

    @staticmethod
    def check_readiness(func):
        def wrapper(*args,**kwargs):
            is_ready = True
            missing = []
            if func.__name__ == "load_data":
                data_can_be_loaded, temp = ModelOps.state.data_can_be_loaded
                is_ready = is_ready and data_can_be_loaded
                missing.append(temp)

            if func.__name__ in ["train", "test"]:
                data_is_loaded, temp = ModelOps.state.data_is_loaded
                missing.append(temp)
                is_ready = is_ready and data_is_loaded

            if not is_ready:
                print(f"Missing following: {missing} from config file")
                return

            func(*args, **kwargs)

        return wrapper

    @classmethod
    @check_readiness
    def load_sample_image(cls):
        ModelOps.load_data()
        verify_dataset(cls.state["training_data"], cls.state["class_names"], cls.state["font_file"], 1)
        verify_dataset(cls.state["test_data"], cls.state["class_names"], cls.state["font_file"], 1)

    @classmethod
    def setup_model(cls, model_path=""):
        # select device
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        cls.state["device"] = device
        print(f"Using {device} device")

        dtype = torch.float32
        # Model
        cls.state["model"] = NeuralNetworkOps(
            device=device,
            dtype=dtype,
            number_of_classes=len(cls.state["class_names"]),
            model_path=model_path
        )

    @classmethod
    # @check_readiness
    def train(cls):
        # Training

        # Create timestamp and directory to store training data and best model
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a color map
        color_map = {'items': [{'label': label, 'color': color} for label, color in zip(cls.state["class_names"], cls.state["colors"])]}

        checkpoint_dir: Path = cls.state["project_dir"] / timestamp
        cls.state["checkpoint_dir"] = checkpoint_dir
        # The model checkpoint path
        checkpoint_path: Path = checkpoint_dir / f"{cls.state['model'].name}.pth"

        ModelOpsUtils.make_config_dir_save_color_map(color_map=color_map,
                                                     checkpoint_dir=checkpoint_dir,
                                                     model_name=cls.state['model'].name,
                                                     dataset_path_name=cls.state["dataset_path"].name,
                                                     )

        # Learning rate for the model
        lr = cls.state["initial_learning_rate"]

        # Number of training epochs
        epochs = cls.state["number_of_epochs"]

        model = cls.state["model"].model

        # AdamW optimizer; includes weight decay for regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Learning rate scheduler; adjusts the learning rate during training
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=lr,
                                                           total_steps=epochs * len(cls.state["dataloader_training_data"]))

        columns = ['epoch', 'train_loss', 'valid_loss', 'learning_rate', 'model_architecture']
        epoch_df = pd.DataFrame({}, columns=columns)
        temp_path = Path(checkpoint_path.parent / 'log.csv')
        epoch_df.to_csv(temp_path, mode='a', index=False, header=True)

        train_loop(model=model,
                   train_dataloader=cls.state["dataloader_training_data"],
                   valid_dataloader=cls.state["dataloader_testing_data"],
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler,
                   device=torch.device(cls.state["device"]),
                   epochs=epochs,
                   checkpoint_path=checkpoint_path,
                   train_sz=cls.state["train_data_size"],
                   batch_size=cls.state["batch_size"],
                   initial_learning_rate=cls.state["initial_learning_rate"],
                   annotation_file_path=cls.state["annotation_file_path"],
                   use_scaler=True)


    @classmethod
    @check_readiness
    def test(cls, checkpoint_dir):
        model = cls.state["model"].model
        model.eval()

        print(checkpoint_dir)

        log_path = checkpoint_dir / "log.csv"

        draw_learning_graph(log_path)

        # Make a copy of the color map in integer format
        int_colors = [tuple(int(c * 255) for c in color) for color in cls.state["colors"]]

        test_model(model=model,
                   device=torch.device(cls.state["device"]),
                   val_keys = cls.state["validation_keys"],
                   image_dict = cls.state["image_dict"],
                   train_data_size = cls.state["train_data_size"],
                   annotation_df = cls.state["annotation_df"],
                   threshold = cls.state["prediction_threshold"],
                   class_names = cls.state["class_names"],
                   int_colors = int_colors,
                   font=cls.state["font_file"],
                   )


    @classmethod
    @check_readiness
    def load_data(cls):
        # Download the font file
        download_file(f"https://fonts.gstatic.com/s/roboto/v30/{cls.state['font_file']}", "./")
        draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=cls.state["font_file"], font_size=10)

        dataset_name = cls.state["dataset_name"]
        dataset_dir = cls.state["dataset_dir"]
        images_subdirectory = cls.state["images_subdirectory"]
        annotation_file_path = cls.state["annotation_file_path"]
        class_names = cls.state["class_names"]
        train_pct = cls.state["train_pct"]

        cls.state["dataset_path"] = Path(f'{dataset_dir}/{dataset_name}')
        dataset_path = cls.state["dataset_path"]
        image_directory = dataset_path / images_subdirectory

        cls.state["project_dir"] = Path(f"./{cls.state['project_name']}/")

        # Creating a Series with the paths and converting it to a DataFrame for display
        dataframe = pd.Series({
            "Image Folder": image_directory,
            "Annotation File": annotation_file_path}).to_frame().style.hide(axis='columns')
        print(dataframe.to_string())

        image_dict, annotation_df, train_keys, val_keys, class_to_idx = DatasetUtils.create_image_and_annotation_dict(
            image_directory,
            annotation_file_path,
            class_names,
            train_pct,
        )

        cls.state["image_dict"] = image_dict
        cls.state["annotation_df"] = annotation_df
        cls.state["colors"] = distinctipy.get_colors(len(class_names))

        cls.state["validation_keys"] = val_keys

        cls.state["training_data"] = COCOLIVECellDataset(train_keys, annotation_df, image_dict, class_to_idx,
                                        train_tfms)
        cls.state["test_data"] = COCOLIVECellDataset(val_keys, annotation_df, image_dict, class_to_idx,
                                    train_tfms)

        temp = pd.Series({
            'Training dataset size:': len(cls.state["training_data"]),
            'Validation dataset size:': len(cls.state["test_data"])}
        ).to_frame()
        print(temp.to_string())


    @classmethod
    def run_epochs(cls):

        # select device
        cls.setup_model()
        device = cls.state["device"]
        model = cls.state["model"].model

        test_inp = tuple(torch.randn(1, 3, 256, 256).to(device))
        print(test_inp)

        summary_df = markdown_to_pandas(f"{get_module_summary(model.eval(), [test_inp])}")

        # # Filter the summary to only contain Conv2d layers and the model
        summary_df = summary_df[summary_df.index == 0]

        temp = summary_df.drop(['In size', 'Out size', 'Contains Uninitialized Parameters?'], axis=1)

        print(temp.to_string())

        # DataLoader
        # Set the training batch size
        bs = cls.state["batch_size"]

        # Set the number of worker processes for loading data.
        # num_workers = multiprocessing.cpu_count() // 2
        num_workers = cls.state["number_of_workers"]

        # Define parameters for DataLoader
        data_loader_params = {
            'batch_size': bs,  # Batch size for data loading
            'num_workers': num_workers,  # Number of subprocesses to use for data loading
            'persistent_workers': True,
            # If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
            'pin_memory': 'cuda' in device,
            # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
            'pin_memory_device': device if 'cuda' in device else '',
            # Specifies the device where the data should be loaded. Commonly set to use the GPU.
            'collate_fn': tuple_batch,
        }

        # Create DataLoader for training data. Data is shuffled for every epoch.
        cls.state["dataloader_training_data"] = DataLoader(cls.state["training_data"], **data_loader_params, shuffle=True)

        # Create DataLoader for validation data. Shuffling is not necessary for validation data.
        cls.state["dataloader_testing_data"] = DataLoader(cls.state["test_data"], **data_loader_params)

        # Print the number of batches in the training and validation DataLoaders
        temp = pd.Series({
            'Number of batches in train DataLoader:': len(cls.state["dataloader_training_data"]),
            'Number of batches in validation DataLoader:': len(cls.state["dataloader_testing_data"])}
        ).to_frame().style.hide(axis='columns')

        cls.train()

        cls.test(cls.state["checkpoint_dir"])

    @classmethod
    def test_on_random_sample(cls):
        cls.load_data()
        cls.setup_model()
        cls.test(cls.state["checkpoint_dir"])

    @classmethod
    def infer(cls):

        cls.setup_model(cls.state["prediction_model_path"])

        model_settings = {
            "class_names": cls.state["class_names"],
            "train_size": cls.state["train_data_size"],
            "device": cls.state["device"],
            "threshold": cls.state["prediction_threshold"]
        }
        segment_image(model=cls.state["model"], image_path=cls.state["image_path"], model_settings=model_settings)

