import random
from pathlib import Path
import pandas as pd

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from cjm_psl_utils.core import download_file

from functools import partial
from distinctipy import distinctipy

from scripts.img_seg_dataset import COCOLIVECellDataset
from scripts.project_utils import get_image_files, train_tfms, verify_dataset

# cleanup later
DATASET_DIR = Path("D:/dev/livecell-dataset/")
DATASET_NAME = 'LIVECell_dataset_2021'
IMAGES_SUBDIRECTORY = r'images/all_images/'
ANNOTATION_FILE_PATH = r"D:\dev\livecell-dataset\LIVECell_dataset_2021\annotations\LIVECell_dataset_size_split\2_train5percent.json"

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
    batch_size: int
    number_of_epochs: int

    training_data: datasets
    test_data: datasets

    training_data_path: str = ""
    testing_data_path: str = ""

    font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

    dataset_name = DATASET_NAME
    dataset_dir = DATASET_DIR
    images_subdirectory = IMAGES_SUBDIRECTORY
    annotation_file_path = ANNOTATION_FILE_PATH

    @staticmethod
    def check_readiness(func):
        def wrapper(*args,**kwargs):
            print("pre test")
            func(*args,**kwargs)
            print("after test")
        return wrapper

    @staticmethod
    def load_settings():

        pass

    @staticmethod
    def load_sample_image():
        print(f"loading data from {ModelOps.training_data_path} and {ModelOps.testing_data_path}")
        ModelOps.load_data()

        print("applying transforms")

        print("displaying data")

    @staticmethod
    def train(dataloader, model, device, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    @staticmethod
    def test(dataloader, model, device, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    @staticmethod
    @check_readiness
    def load_data():
        # Download the font file
        download_file(f"https://fonts.gstatic.com/s/roboto/v30/{ModelOps.font_file}", "./")
        draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=ModelOps.font_file, font_size=10)

        dataset_name = DATASET_NAME
        dataset_dir = DATASET_DIR
        images_subdirectory = IMAGES_SUBDIRECTORY
        annotation_file_path = ANNOTATION_FILE_PATH

        dataset_path = Path(f'{dataset_dir}/{dataset_name}')
        image_directory = dataset_path / images_subdirectory

        project_name = f"pytorch-mask-r-cnn-instance-segmentation"

        project_dir = Path(f"./{project_name}/")

        temp = pd.Series({
            "Project Directory:": project_dir,
            "Dataset Directory:": dataset_dir
        }).to_frame()
        print(temp.to_string)

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

        # Get a list of unique labels in the 'annotation_df' DataFrame
        class_names = annotation_df['labels'].explode().unique().tolist()

        # Prepend a `background` class to the list of class names
        class_names = ['background'] + class_names

        # Create a mapping from class names to class indices
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Generate a list of colors with a length equal to the number of labels
        colors = distinctipy.get_colors(len(class_names))

        # Make a copy of the color map in integer format
        int_colors = [tuple(int(c * 255) for c in color) for color in colors]

        train_pct = 0.8
        # Get the list of image IDs
        img_keys = annotation_df.index.tolist()
        # Shuffle the image IDs
        random.shuffle(img_keys)

        train_split = int(len(img_keys) * train_pct)

        train_keys = img_keys[:train_split]

        ModelOps.training_data = COCOLIVECellDataset(train_keys, annotation_df, image_dict, class_to_idx,
                                        train_tfms)

        print(train_keys)

        verify_dataset(ModelOps.training_data, class_names, ModelOps.font_file)

        # Download test data from open datasets.
        # ModelOps.test_data = datasets.FashionMNIST(
        #     root=ModelOps.training_data_path,
        #     train=False,
        #     download=False,
        #     transform=ToTensor(),
        # )

    @staticmethod
    def run_epochs(model, device):

        batch_size = ModelOps.batch_size
        num_of_epochs = ModelOps.number_of_epochs

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Create data loaders
        train_dataloader = DataLoader(ModelOps.training_data, batch_size=batch_size)
        test_dataloader = DataLoader(ModelOps.test_data, batch_size=batch_size)

        for t in range(num_of_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            ModelOps.train(train_dataloader, model, device, loss_fn, optimizer)
            ModelOps.test(test_dataloader, model, device, loss_fn)
        print("Done!")
