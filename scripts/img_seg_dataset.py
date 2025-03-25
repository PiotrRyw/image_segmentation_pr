from torch.utils.data import Dataset
import torch.utils.data
from torchvision.ops import masks_to_boxes
from torchvision.tv_tensors import Mask, BoundingBoxes
import torchvision.transforms.v2 as tfs

from PIL import Image
import torch

from scripts.project_utils import create_polygon_mask, rle2mask, correct_rotated_masks


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

        # print(annotation)

        mask_imgs = []
        for single_annotation in annotation_data:
            if type(single_annotation) is list:
                mask_imgs.append(create_polygon_mask(image.size, single_annotation[0]))
            elif type(single_annotation) is dict:
                mask_imgs.append(rle2mask(single_annotation)) # function expects a dictionary in a form of
                # {"counts": [list being uncompressed image], "image size": [width, height]}, so we can just pass the
                # full annotation as it is in a correct form already
            else:
                print("bad data - _load_image_and_target in dataset.py")
                raise Exception("wrong data in annotation")

        # something is wrong with some annotations - the dimension is flipped. Drop them and print out
        mask_imgs = correct_rotated_masks(mask_imgs)

        masks = Mask(
            torch.concat([Mask(tfs.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

        # Convert bounding boxes to tensor format
        # taking bboxes FROM MASKS
        bbox_tensor = masks_to_boxes(masks)

        # bbox_list = annotation['bboxes']
        # bbox_tensor = torchvision.ops.box_convert(torch.Tensor(bbox_list), 'xywh', 'xyxy')

        corrected_dims = (image.size[1], image.size[0])
        boxes = BoundingBoxes(bbox_tensor, format='xyxy', canvas_size=corrected_dims)

        # Map labels to their corresponding indices
        annotation_labels = annotation['labels']
        labels = torch.Tensor([self._class_to_idx[label] for label in annotation_labels])
        # Convert the class labels to indices
        labels = labels.to(dtype=torch.int64)

        return image, {'masks': masks, 'boxes': boxes, 'labels': labels}
