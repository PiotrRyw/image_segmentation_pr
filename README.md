Bachelor's thesis on microscopy image segmentation

This project uses PyTorch and COCO datasets (custom dataset and LIVECell dataset) to train a Mask-RCNN model for the task of cell image segmentation.

The goal is to implement a pipeline for loading datasets and training model for the task of instance segmentation of phase contrast microscopy sourced astrocyte images.
After training, the model can be used to segment an image provided in a path in a config file.

The LIVECell dataset is sourced from 
https://github.com/sartorius-research/LIVECell
