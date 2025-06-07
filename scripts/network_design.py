import torch
import torchvision
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class NeuralNetworkOpsBaseClass:
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn_v2()
        self.model.name = "base_class_maskrcnn_resnet50_fpn_v2"


class NeuralNetworkOps(NeuralNetworkOpsBaseClass):
    def __init__(self, device, dtype, number_of_classes, model_name=""):
        super().__init__()
        # Model

        # Initialize a Mask R-CNN model (with or without) pretrained weights
        self.model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        if model_name == "":
            self.model.name = 'maskrcnn_resnet50_fpn_v2'
        else:
            self.model.name = model_name
        # Get the number of input features for the classifier
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        # Get the number of output channels for the Mask Predictor
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=number_of_classes)

        # Replace the mask predictor
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                           num_classes=number_of_classes)

        # Set the model's device and data type
        self.model.to(device=device, dtype=dtype)


    def load_state_from_file(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded state from {model_path}")
        except FileNotFoundError:
            print(f"No model file exists")

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
