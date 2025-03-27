import numpy as np
import pandas as pd
import torch
from PIL import Image
from cjm_pil_utils.core import resize_img
from cjm_pytorch_utils.core import move_data_to_device, tensor_to_pil
from distinctipy import distinctipy
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import Mask, BoundingBoxes
from torchvision.utils import draw_segmentation_masks
import torch.nn.functional as F

def segment_image(model, image_path, model_settings):
    class_names = model_settings["class_names"]
    device = model_settings["device"]
    threshold = model_settings["threshold"]

    test_img = Image.open(image_path).convert('RGB')
    input_img = resize_img(test_img, target_sz=model_settings["train_size"], divisor=1)
    # Calculate the scale between the source image and the resized image
    min_img_scale = min(test_img.size) / min(input_img.size)

    # Print the prediction data as a Pandas DataFrame for easy formatting
    temp = pd.Series({
        "Source Image Size:": test_img.size,
        "Input Dims:": input_img.size,
        "Min Image Scale:": min_img_scale,
        "Input Image Size:": input_img.size
    }).to_frame().style.hide(axis='columns')
    print(temp.to_string())

    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(len(class_names))

    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c * 255) for c in color) for color in colors]

    model.model.eval()
    model.model.to(device)

    input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(input_img)[
        None].to(device)

    with torch.no_grad():
        model_output = model.model(input_tensor)

    # Move model output to the CPU
    model_output = move_data_to_device(model_output, 'cpu')

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
    pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=input_img.size[::-1])

    # if no predictions were made, skip and write warning message
    if pred_masks.size(dim=0) == 0:
        print("no predictions for the image [shown]:")
        input_img.show()
    else:
        pred_masks = torch.concat([Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])

        # Convert the test images to a tensor
        img_tensor = transforms.PILToTensor()(input_img)

        # Annotate the test image with the predicted segmentation masks
        annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.3)
        pred_colors = [int_colors[i] for i in [class_names.index(label) for label in pred_labels]]
        # Annotate the test image with the predicted labels and bounding boxes
        # annotated_tensor = draw_bboxes(
        #     image=annotated_tensor,
        #     boxes=pred_bboxes,
        #     labels=[f"{label}\n{prob * 100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
        #     colors=pred_colors,
        #     font=font_file,
        #     font_size=10
        # )

        # Print the prediction data as a Pandas DataFrame for easy formatting
        pd.Series({
            "Predicted BBoxes:": [f"{label}:{bbox}" for label, bbox in
                                  zip(pred_labels, pred_bboxes.round(decimals=3).numpy())],
            "Confidence Scores:": [f"{label}: {prob * 100:.2f}%" for label, prob in zip(pred_labels, pred_scores)]
        }).to_frame().style.hide(axis='columns')

        image_segmented = tensor_to_pil(annotated_tensor)
        Image.fromarray(np.hstack((np.array(input_img), np.array(image_segmented)))).show()
