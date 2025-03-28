import json
import random
import time
import os
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
from torchvision.ops import masks_to_boxes
from torchvision.tv_tensors import Mask, BoundingBoxes
import torchvision.transforms.v2 as tfs

from PIL import Image
import torch
from cjm_pytorch_utils.core import tensor_to_pil

from scripts.project_utils import create_polygon_mask, rle2mask, correct_rotated_masks, get_image_files, train_tfms, \
    correct_rotated_masks_old


class COCOLIVECellDataset(Dataset):
    """
    A PyTorch Dataset class for COCO-style instance segmentation.

    This class is designed to handle datasets for instance segmentation tasks, specifically
    formatted in the style of COCO (Common Objects in Context) annotations. It supports
    loading images along with their corresponding segmentation masks and bounding boxes.

    Attributes:
    _img_keys : list
        List of image keys (identifiers).
    _annotation_df : pandas.DataFrame
        DataFrame containing annotations for images.
    _img_dict : dict
        Dictionary mapping image keys to their file paths.
    _class_to_idx : dict
        Dictionary mapping class names to class indices.
    _transforms : torchvision.transforms (optional)
        Transformations to be applied to the images and targets.

    Methods:
    __init__(self, img_keys, annotation_df, img_dict, class_to_idx, transforms=None):
        Initializes the dataset with image keys, annotations, image dictionary,
        class mappings, and optional transforms.
    __len__(self):
        Returns the total number of items in the dataset.
    __getitem__(self, index):
        Retrieves an image and its corresponding target (masks, boxes, labels) by index.
    _load_image_and_target(self, annotation):
        Loads an image and its corresponding target data based on the providedannotation.

    """

    def __init__(self, img_keys, annotation_df, img_dict, class_to_idx, defined_transforms=None):
        """
        Initializes the COCOInstSegDataset instance.

        Parameters:
            img_keys (list): List of image keys.
            annotation_df (DataFrame): DataFrame containing image annotations.
            img_dict (dict): Dictionary mapping image keys to file paths.
            class_to_idx (dict): Dictionary mapping class names to indices.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()

        self._img_keys = img_keys  # List of image keys
        self._annotation_df = annotation_df  # DataFrame containing annotations
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._class_to_idx = class_to_idx  # Dictionary mapping class names to class indices
        self._transforms = defined_transforms  # Image transforms to be applied

    def __len__(self):
        # Returns the number of items in the dataset
        return len(self._img_keys)

    def __getitem__(self, index):
        # Retrieves an image and its corresponding target by index
        img_key = self._img_keys[index]
        annotation = self._annotation_df.loc[img_key]
        image, target = self._load_image_and_target(annotation)

        # Apply transformations if any
        if self._transforms:
            image, target = self._transforms(image, target)

        return image, target

    def _load_image_and_target(self, annotation):
        # Loads an image and its corresponding target data (masks, boxes, labels)
        filepath = self._img_dict[annotation.name]
        image = Image.open(filepath).convert('RGB')

        # Process segmentation polygons
        annotation_data = annotation['segmentation']

        mask_imgs_cpm = []
        mask_imgs_rle = []
        for single_annotation in annotation_data:
            if type(single_annotation) is list:
                mask_imgs_cpm.append(create_polygon_mask(image.size, single_annotation[0]))
            elif type(single_annotation) is dict:
                mask_imgs_rle.append(rle2mask(single_annotation)) # function expects a dictionary in a form of
                # {"counts": [list being uncompressed image], "image size": [width, height]}, so we can just pass the
                # full annotation as it is in a correct form already
            else:
                raise RuntimeError(f"Wrong annotation data for file with id {filepath}")

        correct_rotated_masks(mask_imgs_rle)

        mask_imgs = mask_imgs_cpm
        mask_imgs.extend(mask_imgs_rle)
        # something is wrong with some annotations - the dimension is flipped. Drop them and print out
        # correct_rotated_masks(mask_imgs)
        masks = Mask(
            torch.concat([Mask(tfs.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

        # Convert bounding boxes to tensor format
        # taking bboxes FROM MASKS
        bbox_tensor = masks_to_boxes(masks)

        corrected_dims = (image.size[1], image.size[0])
        boxes = BoundingBoxes(bbox_tensor, format='xyxy', canvas_size=corrected_dims)

        # Map labels to their corresponding indices
        annotation_labels = annotation['labels']
        labels = torch.Tensor([self._class_to_idx[label] for label in annotation_labels])
        # Convert the class labels to indices
        labels = labels.to(dtype=torch.int64)

        return image, {'masks': masks, 'boxes': boxes, 'labels': labels}


class DatasetUtils:
    @classmethod
    def find_cache(cls):
        max_it = 2
        while not os.path.exists("cache") and max_it > 0:
            os.chdir("..")
            max_it -= 1

        os.chdir("cache/annotation_dataframe")



    @classmethod
    def check_cached(cls, file_path):
        cls.find_cache()
        with open("cached.json", "r") as file:
            temp = json.load(file)
        if file_path in temp:
            return temp[file_path]
        else:
            return None

    @classmethod
    def cache_result(cls, original_file_path, data: pd.DataFrame):
        head, cached_data_path = os.path.split(original_file_path)
        absolute_path = os.path.join(os.getcwd(), cached_data_path)

        cls.find_cache()

        with open(absolute_path, "w") as file:
            data.to_json(file)

        temp = None
        with open("cached.json", "r") as file:
            temp = json.load(file)


        temp[original_file_path] = absolute_path
        with open("cached.json", "w") as file:
            json.dump(temp, file)


    @classmethod
    def create_dataframe(cls, annotation_file_df):
        categories_df = annotation_file_df['categories'].dropna().apply(pd.Series)
        categories_df.set_index('id', inplace=True)
        # print(categories_df)

        # Extract and transform the 'images' section of the data
        # This DataFrame contains image details like file name, height, width, and image ID
        images_df = annotation_file_df['images'].to_frame()['images'].apply(pd.Series)[
            ['file_name', 'height', 'width', 'id']]

        print(f"Extracted image file data")

        # Extract and transform the 'annotations' section of the data
        # This DataFrame contains annotation details like image ID, segmentation points, bounding box, and category ID
        annotations_df = annotation_file_df['annotations'].to_frame()['annotations'].apply(pd.Series)[
            ['image_id', 'segmentation', 'bbox', 'category_id']]
        pd.options.display.max_columns = None
        pd.options.display.width = 0

        # Map 'category_id' in annotations DataFrame to category name using categories DataFrame
        annotations_df['label'] = annotations_df['category_id'].apply(lambda x: categories_df.loc[x]['name'])

        # Merge annotations DataFrame with images DataFrame on their image ID
        annotation_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')

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

        return annotations_df

    @classmethod
    def _private_create_image_and_annotation_dict(cls,
                                                  image_directory: Path,
                                                  annotation_file_path: str,
                                                  class_names: List[str],
                                                  train_pct: float,
                                                  performance_check = False
                                                  ):

        app_working_directory = os.getcwd()

        _time_start = time.time()
        # Get all image files in the 'img_dir' directory
        image_dict = {
            file.stem: file  # Create a dictionary that maps file names to file paths
            for file in get_image_files(image_directory)  # Get a list of image files in the image directory
        }
        print(f"Number of Images: {len(image_dict)}")

        temp = cls.check_cached(annotation_file_path)
        if temp:  # is not None
            with open(temp, "r") as file:
                annotation_df = pd.read_json(file)
            print(f"Loaded cached file {temp}")
        else:
            # Read the JSON file into a DataFrame, assuming the JSON is oriented by index
            annotation_file_df = pd.read_json(annotation_file_path, orient='index').transpose()
            # print(annotation_file_df.head())
            print(f"Loaded file {annotation_file_path}")
            annotation_df = cls.create_dataframe(annotation_file_df)

        # Create a mapping from class names to class indices
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Get the list of image IDs
        img_keys: List = annotation_df.index.tolist()

        # Shuffle the image IDs
        random.shuffle(img_keys)

        train_split = int(len(img_keys) * train_pct)
        train_keys = img_keys[:train_split]
        val_keys = img_keys[train_split:]
        print("Loaded image data and annotation data")

        if performance_check:
            _time_end = time.time()
            print(f"Finished the rest - time: {_time_end - _time_start}")
            print(annotation_df.info(memory_usage='deep'))

        cls.cache_result(annotation_file_path, annotation_df)

        os.chdir(app_working_directory)

        return image_dict, annotation_df, train_keys, val_keys, class_to_idx

    @classmethod
    def create_image_and_annotation_dict(cls,
                                         image_directory: Path,
                                         annotation_file_path: str,
                                         class_names: List[str],
                                         train_pct: float,
                                         ):
        return cls._private_create_image_and_annotation_dict(image_directory,
                                                             annotation_file_path,
                                                             class_names,
                                                             train_pct,
                                                             )


def test():
    # annotation_file_path = "D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/0_train2percent.json"
    annotation_file_path = "D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/2_train5percent.json"
    # annotation_file_path = "D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/4_train50percent.json"


    image_dict, annotation_df, train_keys, val_keys, class_to_idx = DatasetUtils._private_create_image_and_annotation_dict(
        image_directory=Path("D:/dev/livecell-dataset/LIVECell_dataset_2021/images/all_images"),
        annotation_file_path=annotation_file_path,
        class_names=["background", "cell"],
        train_pct=0.8,
        performance_check=True,
    )

    dataset = COCOLIVECellDataset(train_keys, annotation_df, image_dict, class_to_idx,
                                                     train_tfms)

    dataset.__getitem__(0)

    # Choose a random item from the validation set
    file_id = random.choice(val_keys)
    test_file = image_dict[file_id]
    test_img = Image.open(test_file).convert('RGB')
    test_img.show()


def test_dataloader():
    annotation_file_path = "D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/2_train5percent.json"
    # annotation_file_path = "D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/4_train50percent.json"

    image_dict, annotation_df, train_keys, val_keys, class_to_idx = DatasetUtils._private_create_image_and_annotation_dict(
        image_directory=Path("D:/dev/livecell-dataset/LIVECell_dataset_2021/images/all_images"),
        annotation_file_path=annotation_file_path,
        class_names=["background", "cell"],
        train_pct=0.8,
        performance_check=True,
    )

    dataset = COCOLIVECellDataset(train_keys, annotation_df, image_dict, class_to_idx,
                                  train_tfms)
    _time_start = time.time()
    for i in range(50):
        image, target = dataset.__getitem__(i)
    _time_end = time.time()
    print(f"Elapsed time: {_time_end - _time_start}")
    #
    # img = tensor_to_pil(image)
    # img.show()

if __name__ == "__main__":
    # test()
    test_dataloader()
