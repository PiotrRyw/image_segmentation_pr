import json
import random
import datetime
from pathlib import Path
import pandas as pd

import torch
from jinja2 import ModuleLoader
from torch import nn
from torchvision import datasets
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from cjm_pandas_utils.core import markdown_to_pandas
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchtnt.utils import get_module_summary

from cjm_psl_utils.core import download_file

from functools import partial
from distinctipy import distinctipy

from scripts.img_seg_dataset import COCOLIVECellDataset
from scripts.project_utils import get_image_files, train_tfms, verify_dataset, tuple_batch, draw_learning_graph
from scripts.test_model import test_model
from scripts.train import train_loop


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ModelOps:
    batch_size: int = None
    number_of_epochs: int = None
    number_of_workers: int = None
    train_data_size: int = None

    training_data: datasets = None
    test_data: datasets = None

    dataloader_training_data = None
    dataloader_testing_data = None

    font_file: str = None

    colors = None
    initial_learning_rate: float = None

    dataset_name = None
    dataset_dir = None
    images_subdirectory = None
    annotation_file_path = None
    project_dir = None

    model_path: str = None
    project_name: str = None

    train_pct: int = None

    class_names: str = None
    dataset_path = None

    validation_keys = None

    annotation_df = None

    model = None
    device = None

    image_dict = None

    prediction_threshold: float = None
    checkpoint_dir = None

    @staticmethod
    def check_readiness(func):
        def wrapper(*args,**kwargs):
            requirements = {
                "batch_size": ModelOps.batch_size,
                "number_of_epochs": ModelOps.number_of_epochs,

                "dataset_name": ModelOps.dataset_name,
                "dataset_dir": ModelOps.dataset_dir,
                "images_subdirectory": ModelOps.images_subdirectory,
                "annotation_file_path": ModelOps.annotation_file_path,
                "project_name": ModelOps.project_name,
                "train_pct": ModelOps.train_pct,
            }

            if func.__name__ in ["train", "test"]:
                requirements["training_data"] = ModelOps.training_data
                requirements["test_data"] = ModelOps.test_data


            missing_attributes = []
            for entity in requirements.keys():
                if requirements[entity] is None:
                    missing_attributes.append(entity)

            if len(missing_attributes):
                print(f"Missing following: {missing_attributes}")
                return

            func(*args, **kwargs)

        return wrapper

    @staticmethod
    @check_readiness
    def load_sample_image():
        ModelOps.load_data()
        verify_dataset(ModelOps.training_data, ModelOps.class_names, ModelOps.font_file, 1)
        verify_dataset(ModelOps.test_data, ModelOps.class_names, ModelOps.font_file, 1)



    @staticmethod
    @check_readiness
    def train():
        # Training
        # Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a directory to store the checkpoints if it does not already exist

        checkpoint_dir = Path(ModelOps.project_dir / f"{timestamp}")
        ModelOps.checkpoint_dir = checkpoint_dir

        # Create the checkpoint directory if it does not already exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # The model checkpoint path
        checkpoint_path = checkpoint_dir / f"{ModelOps.model.name}.pth"

        print(checkpoint_path)

        # Create a color map and write it to a JSON file
        color_map = {'items': [{'label': label, 'color': color} for label, color in zip(ModelOps.class_names, ModelOps.colors)]}
        with open(f"{checkpoint_dir}/{ModelOps.dataset_path.name}-colormap.json", "w") as file:
            json.dump(color_map, file)

        # Print the name of the file that the color map was written to
        print(f"{checkpoint_dir}/{ModelOps.dataset_path.name}-colormap.json")
        # Learning rate for the model
        lr = ModelOps.initial_learning_rate

        # Number of training epochs
        epochs = ModelOps.number_of_epochs

        # AdamW optimizer; includes weight decay for regularization
        optimizer = torch.optim.AdamW(ModelOps.model.parameters(), lr=lr)

        # Learning rate scheduler; adjusts the learning rate during training
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=lr,
                                                           total_steps=epochs * len(ModelOps.dataloader_training_data))

        # If the device is a GPU, empty the cache
        columns = ['epoch', 'train_loss', 'valid_loss', 'learning_rate', 'model_architecture']
        epoch_df = pd.DataFrame({}, columns=columns)
        temp_path = Path(checkpoint_path.parent / 'log.csv')
        epoch_df.to_csv(temp_path, mode='a', index=False, header=True)

        train_loop(model=ModelOps.model,
                   train_dataloader=ModelOps.dataloader_training_data,
                   valid_dataloader=ModelOps.dataloader_testing_data,
                   optimizer=optimizer,
                   lr_scheduler=lr_scheduler,
                   device=torch.device(ModelOps.device),
                   epochs=epochs,
                   checkpoint_path=checkpoint_path,
                   train_sz=ModelOps.train_data_size,
                   batch_size=ModelOps.batch_size,
                   initial_learning_rate=ModelOps.initial_learning_rate,
                   annotation_file_path=ModelOps.annotation_file_path,
                   use_scaler=True)


    @staticmethod
    @check_readiness
    def test(checkpoint_dir):
        model = ModelOps.model
        model.eval()

        print(checkpoint_dir)

        log_path = checkpoint_dir / "log.csv"

        draw_learning_graph(log_path)

        # Make a copy of the color map in integer format
        int_colors = [tuple(int(c * 255) for c in color) for color in ModelOps.colors]

        test_model(model=model,
                   device=ModelOps.device,
                   val_keys = ModelOps.validation_keys,
                   image_dict = ModelOps.image_dict,
                   train_data_size = ModelOps.train_data_size,
                   annotation_df = ModelOps.annotation_df,
                   threshold = ModelOps.prediction_threshold,
                   class_names = ModelOps.class_names,
                   int_colors = int_colors,
                   font=ModelOps.font_file,
                   )


    @staticmethod
    @check_readiness
    def load_data():
        # Download the font file
        download_file(f"https://fonts.gstatic.com/s/roboto/v30/{ModelOps.font_file}", "./")
        draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=ModelOps.font_file, font_size=10)

        dataset_name = ModelOps.dataset_name
        dataset_dir = ModelOps.dataset_dir
        images_subdirectory = ModelOps.images_subdirectory
        annotation_file_path = ModelOps.annotation_file_path

        ModelOps.dataset_path = Path(f'{dataset_dir}/{dataset_name}')
        dataset_path = ModelOps.dataset_path
        image_directory = dataset_path / images_subdirectory

        ModelOps.project_dir = Path(f"./{ModelOps.project_name}/")

        # Creating a Series with the paths and converting it to a DataFrame for display
        dataframe = pd.Series({
            "Image Folder": image_directory,
            "Annotation File": annotation_file_path}).to_frame().style.hide(axis='columns')
        print(dataframe.to_string())

        # Get all image files in the 'img_dir' directory
        image_dict = {
            file.stem: file  # Create a dictionary that maps file names to file paths
            for file in get_image_files(image_directory)  # Get a list of image files in the image directory
        }
        ModelOps.image_dict = image_dict

        # Print the number of image files
        print(f"Number of Images: {len(image_dict)}")

        # Read the JSON file into a DataFrame, assuming the JSON is oriented by index
        annotation_file_df = pd.read_json(annotation_file_path, orient='index').transpose()
        #print(annotation_file_df.head())

        categories_df = annotation_file_df['categories'].dropna().apply(pd.Series)
        categories_df.set_index('id', inplace=True)
        #print(categories_df)

        # Extract and transform the 'images' section of the data
        # This DataFrame contains image details like file name, height, width, and image ID
        images_df = annotation_file_df['images'].to_frame()['images'].apply(pd.Series)[
            ['file_name', 'height', 'width', 'id']]
        print(images_df.head())

        # Extract and transform the 'annotations' section of the data
        # This DataFrame contains annotation details like image ID, segmentation points, bounding box, and category ID
        annotations_df = annotation_file_df['annotations'].to_frame()['annotations'].apply(pd.Series)[
            ['image_id', 'segmentation', 'bbox', 'category_id']]
        pd.options.display.max_columns = None
        pd.options.display.width = 0
        print(annotations_df.head())

        # Map 'category_id' in annotations DataFrame to category name using categories DataFrame
        annotations_df['label'] = annotations_df['category_id'].apply(lambda x: categories_df.loc[x]['name'])
        print(annotations_df.head())

        # Merge annotations DataFrame with images DataFrame on their image ID
        annotation_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')
        print(annotation_df.head())

        # Remove old 'id' column post-merge
        annotation_df.drop('id', axis=1, inplace=True)

        # Extract the image_id from the file_name (assuming file_name contains the image_id)
        annotation_df['image_id'] = annotation_df['file_name'].apply(lambda x: x.split('.')[0])

        # Set 'image_id' as the index for the DataFrame
        annotation_df.set_index('image_id', inplace=True)

        # Group the data by 'image_id' and aggregate information
        annotation_df = annotation_df.groupby('image_id').agg({
            'segmentation': list,
            'bbox': list,
            'category_id': list,
            'label': list,
            'file_name': 'first',
            'height': 'first',
            'width': 'first'
        })

        # Rename columns for clarity
        annotation_df.rename(columns={'bbox': 'bboxes', 'label': 'labels'}, inplace=True)

        print(annotation_df.head())
        ModelOps.annotation_df = annotation_df

        # Get a list of unique labels in the 'annotation_df' DataFrame
        class_names = annotation_df['labels'].explode().unique().tolist()

        # Prepend a `background` class to the list of class names
        class_names = ['background'] + class_names

        ModelOps.class_names = class_names

        # Create a mapping from class names to class indices
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Generate a list of colors with a length equal to the number of labels
        ModelOps.colors = distinctipy.get_colors(len(class_names))

        # Make a copy of the color map in integer format
        int_colors = [tuple(int(c * 255) for c in color) for color in ModelOps.colors]

        # Get the list of image IDs
        img_keys = annotation_df.index.tolist()
        # Shuffle the image IDs
        random.shuffle(img_keys)

        train_split = int(len(img_keys) * ModelOps.train_pct)

        train_keys = img_keys[:train_split]
        val_keys = img_keys[train_split:]
        ModelOps.validation_keys = val_keys

        ModelOps.training_data = COCOLIVECellDataset(train_keys, annotation_df, image_dict, class_to_idx,
                                        train_tfms)
        ModelOps.test_data = COCOLIVECellDataset(val_keys, annotation_df, image_dict, class_to_idx,
                                    train_tfms)

        temp = pd.Series({
            'Training dataset size:': len(ModelOps.training_data),
            'Validation dataset size:': len(ModelOps.test_data)}
        ).to_frame()
        print(temp.to_string())


    @staticmethod
    def run_epochs():

        # select device
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        ModelOps.device = device
        print(f"Using {device} device")

        dtype = torch.float32
        # Model
        # Initialize a Mask R-CNN model with pretrained weights
        model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
        ModelOps.model = model
        # model = maskrcnn_resnet50_fpn_v2()

        # Get the number of input features for the classifier
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Get the number of output channels for the Mask Predictor
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(ModelOps.class_names))

        # Replace the mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                           num_classes=len(ModelOps.class_names))

        # Set the model's device and data type
        model.to(device=device, dtype=dtype)

        # Add attributes to store the device and model name for later reference
        model.device = device
        model.name = 'maskrcnn_resnet50_fpn_v2'

        test_inp = torch.randn(1, 3, 256, 256).to(device)

        summary_df = markdown_to_pandas(f"{get_module_summary(model.eval(), [test_inp])}")

        # # Filter the summary to only contain Conv2d layers and the model
        summary_df = summary_df[summary_df.index == 0]

        temp = summary_df.drop(['In size', 'Out size', 'Contains Uninitialized Parameters?'], axis=1)

        print(temp.to_string())

        # DataLoader
        # Set the training batch size
        bs = ModelOps.batch_size

        # Set the number of worker processes for loading data.
        # num_workers = multiprocessing.cpu_count() // 2
        num_workers = ModelOps.number_of_workers

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
        ModelOps.dataloader_training_data = DataLoader(ModelOps.training_data, **data_loader_params, shuffle=True)

        # Create DataLoader for validation data. Shuffling is not necessary for validation data.
        ModelOps.dataloader_testing_data = DataLoader(ModelOps.test_data, **data_loader_params)

        # Print the number of batches in the training and validation DataLoaders
        temp = pd.Series({
            'Number of batches in train DataLoader:': len(ModelOps.dataloader_training_data),
            'Number of batches in validation DataLoader:': len(ModelOps.dataloader_testing_data)}
        ).to_frame().style.hide(axis='columns')

        ModelOps.train()

        ModelOps.test(ModelOps.checkpoint_dir)
