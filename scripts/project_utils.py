import csv
import random
from pathlib import Path  # For working with file paths
import hashlib
from glob import glob
import numpy as np  # For working with arrays
from PIL import Image, ImageDraw  # For working with images
from distinctipy import distinctipy
from pycocotools import mask
import pandas as pd
import matplotlib.pyplot as plt
from cjm_torchvision_tfms.core import CustomRandomIoUCrop, ResizeMax, PadSquare
import torchvision.transforms.v2 as transforms
import torch
from torchvision.utils import draw_segmentation_masks
from cjm_pytorch_utils.core import tensor_to_pil

TRAIN_SZ = 256


def rle2mask(uncompressed_rle):
    """
    :param uncompressed_rle: segmentation mask encoded in uncompressed RLE
    :return: segmentation mask in PIL Image
    """

    width = uncompressed_rle["size"][0]
    height = uncompressed_rle["size"][1]

    #  frPyObjects( [pyObjects], h, w )
    binary_rle = mask.frPyObjects(uncompressed_rle, height, width)
    decoded_value = mask.decode(binary_rle)

    pil_mask = Image.fromarray(decoded_value * 255)

    return pil_mask


def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)

    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img

class OrientationCorrection:
    def __init__(self):
        self.height: int | None = None
        self.width: int | None = None

    def define_height_width(self, height, width):
        self.height = height
        self.width = width

    @staticmethod
    def check_if_ready(func):
        def wrapper(self, *args, **kwargs):

            assert isinstance(self.height, int)
            assert isinstance(self.width, int)

            func(self, *args, **kwargs)

        return wrapper

    @check_if_ready
    def correct_rotated_masks(self, masks_list):
        for i in range(len(masks_list)):
            if masks_list[i].height == self.height and masks_list[i].width == self.width:
                continue
            else:
                mask_img = masks_list[i]
                mask_img = mask_img.rotate(90, expand=True)
                mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
                masks_list[i] = mask_img

    @check_if_ready
    def correct_rotated_masks_old(self, masks_list):
        good_list = []
        for mask_img in masks_list:
            if mask_img.height == self.height and mask_img.width == self.width:
                good_list.append(mask_img)
            else:
                mask_img = mask_img.rotate(90, expand=True)
                mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
                good_list.append(mask_img)

        return good_list


def get_image_files(img_dir: Path,  # The directory to search for image files
                    img_fmts=None  # The list of image formats to search for
                    ):
    """
    Get all the image files in the given directory.

    Returns:
    img_paths (list): A list of pathlib.Path objects representing the image files
    """
    if img_fmts is None:
        img_fmts = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
    img_paths = []

    # Use the glob module to search for image files with specified formats
    for fmt in img_fmts:
        img_paths.extend(glob(f'{img_dir}/*.{fmt}'))
    # Convert the file paths to pathlib.Path objects
    img_paths = [Path(path) for path in img_paths]

    return img_paths


# Create a RandomIoUCrop object
iou_crop = CustomRandomIoUCrop(min_scale=0.5,
                               max_scale=1,
                               min_aspect_ratio=0.75,
                               max_aspect_ratio=1.5,
                               sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                               trials=60,
                               jitter_factor=0.1)

# Create a `ResizeMax` object
resize_max = ResizeMax(max_sz=TRAIN_SZ)

# Create a `PadSquare` object
pad_square = PadSquare(shift=True, fill=135)

data_aug_tfms = transforms.Compose(
    transforms=[
        iou_crop,
        transforms.ColorJitter(
                brightness=(0.9, 1.1),
                contrast=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=90)
    ],
)

# Compose transforms to resize and pad in\put images
resize_pad_tfm = transforms.Compose([
    resize_max,
    pad_square,
    transforms.Resize([TRAIN_SZ] * 2, antialias=True)
])

# Compose transforms to sanitize bounding boxes and normalize input data
final_tfms = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.SanitizeBoundingBoxes(),
])

# Define the transformations for training and validation datasets
train_tfms = transforms.Compose([
    data_aug_tfms,
    resize_pad_tfm,
    final_tfms
])

valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])


def draw_learning_graph(data):
    all_data = pd.read_csv(data, index_col=0)
    training_data = all_data[["train_loss", "valid_loss"]]
    training_data.plot()
    plt.show()

def prepare_learning_graph(data):
    all_data = pd.read_csv(data, index_col=0)
    training_data = all_data[["train_loss", "valid_loss"]]
    return training_data

def tuple_batch(batch):
    return tuple(zip(*batch))


def verify_dataset(dataset, clss_names, font_file, ite=1):
    # Print the number of samples in the training dataset
    pd.Series({
        'Training dataset size:': len(dataset),
    }).to_frame().style.hide(axis='columns')

    for i in range(ite):
        dataset_sample = dataset[random.randint(0, len(clss_names) - 1)]

        # Generate a list of colors with a length equal to the number of labels
        colors = distinctipy.get_colors(len(clss_names))

        # Make a copy of the color map in integer format
        int_colors = [tuple(int(c * 255) for c in color) for color in colors]

        # Get colors for dataset sample
        sample_colors = [int_colors[int(i.item())] for i in dataset_sample[1]['labels']]

        # Annotate the sample image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=(dataset_sample[0] * 255).to(dtype=torch.uint8),
            masks=dataset_sample[1]['masks'],
            alpha=0.3,
            # colors=sample_colors
        )

        image = tensor_to_pil(annotated_tensor)
        image.show()
