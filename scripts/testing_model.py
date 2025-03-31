import random
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from cjm_pil_utils.annotation import draw_bboxes
from cjm_pil_utils.core import resize_img
from torchvision.tv_tensors import Mask, BoundingBoxes
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_segmentation_masks
from cjm_pytorch_utils.core import tensor_to_pil, move_data_to_device
import torch.nn.functional as F

from scripts.project_utils import create_polygon_mask, rle2mask, OrientationCorrection

def test_model(model,
               device,
               val_keys,
               image_dict,
               train_data_size,
               annotation_df,
               threshold,
               class_names,
               int_colors,
               font,
               orientation_corr,
               ):

    # Choose a random item from the validation set
    file_id = random.choice(val_keys)

    # Retrieve the image file path associated with the file ID
    test_file = image_dict[file_id]

    # Open the test file
    test_img = Image.open(test_file).convert('RGB')

    # Resize the test image
    input_img = resize_img(test_img, target_sz=train_data_size, divisor=1)

    # Calculate the scale between the source image and the resized image
    min_img_scale = min(test_img.size) / min(input_img.size)

    # GROUND TRUTH:

    # Print the prediction data as a Pandas DataFrame for easy formatting
    temp = pd.Series({
        "Source Image Size:": test_img.size,
        "Input Dims:": input_img.size,
        "Min Image Scale:": min_img_scale,
        "Input Image Size:": input_img.size
    }).to_frame().style.hide(axis='columns')
    print(temp.to_string())

    # Extract the polygon points for segmentation mask
    annotation_data = annotation_df.loc[file_id]['segmentation']

    # Generate mask images from polygons/rle
    mask_imgs = []
    for single_annotation in annotation_data:
        if type(single_annotation) is list:
            mask_imgs.append(create_polygon_mask(test_img.size, single_annotation[0]))
        elif type(single_annotation) is dict:
            mask_imgs.append(rle2mask(single_annotation))  # function expects a dictionary in a form of
            # {"counts": [list being uncompressed image], "image size": [width, height]}, so we can just pass the
            # full annotation as it is in a correct form already
        else:
            print("bad data - _load_image_and_target in dataset.py")
            raise Exception("wrong data in annotation")

    # something is wrong with some annotations - the dimension is flipped. Drop them and print out
    orientation_corr.correct_rotated_masks(mask_imgs)

    # Convert mask images to tensors
    target_masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])

    # Get the target labels and bounding boxes

    target_labels = annotation_df.loc[file_id]['labels']
    canvas_dims = (test_img.size[1], test_img.size[0])
    target_bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(target_masks), format='xyxy',
                                  canvas_size=canvas_dims)

    # Convert the test images to a tensor
    img_tensor = transforms.PILToTensor()(test_img)

    # Annotate the test image with the truth segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=target_masks, alpha=0.3)

    image1 = tensor_to_pil(annotated_tensor)
    # PREDICTION:

    # Ensure the model and input data are on the same device
    model.to(device)
    input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(input_img)[
        None].to(device)

    # Make a prediction with the model
    with torch.no_grad():
        model_output = model(input_tensor)

    # Move model output to the CPU
    model_output = move_data_to_device(model_output, 'cpu')
    # model_output = move_data_to_device(model_output, device)
    print(model_output)
    print(model_output[0]['scores'])
    # Filter the output based on the confidence threshold
    scores_mask = model_output[0]['scores'] > threshold

    # Scale the predicted bounding boxes
    pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask] * min_img_scale, format='xyxy',
                                canvas_size=input_img.size[::-1])

    # Get the class names for the predicted label indices
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]

    # Extract the confidence scores
    pred_scores = model_output[0]['scores']

    # Scale and stack the predicted segmentation masks
    pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=test_img.size[::-1])

    # if no predictions were made, skip and write warning message
    if pred_masks.size(dim=0) == 0:
        print("no predictions for the image [shown]:")
        image1.show()
    else:
        pred_masks = torch.concat([Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])
        # Get the annotation colors for the targets and predictions
        target_colors=[int_colors[i] for i in [class_names.index(label) for label in target_labels]]
        pred_colors=[int_colors[i] for i in [class_names.index(label) for label in pred_labels]]

        # Convert the test images to a tensor
        img_tensor = transforms.PILToTensor()(test_img)

        # Annotate the test image with the predicted segmentation masks
        annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.3)

        # Annotate the test image with the predicted labels and bounding boxes
        # annotated_tensor = draw_bboxes(
        #     image=annotated_tensor,
        #     boxes=pred_bboxes,
        #     labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
        #     colors=pred_colors,
        #     font=font
        # )

        # Print the prediction data as a Pandas DataFrame for easy formatting
        pd.Series({
            "Target BBoxes:": [f"{label}:{bbox}" for label, bbox in zip(target_labels, np.round(target_bboxes.numpy(), decimals=3))],
            "Predicted BBoxes:": [f"{label}:{bbox}" for label, bbox in zip(pred_labels, pred_bboxes.round(decimals=3).numpy())],
            "Confidence Scores:": [f"{label}: {prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)]
        }).to_frame().style.hide(axis='columns')

        image2 = tensor_to_pil(annotated_tensor)

        Image.fromarray(np.hstack((np.array(image1), np.array(image2)))).show()