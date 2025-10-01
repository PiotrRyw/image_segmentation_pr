Bachelor's thesis on microscopy image segmentation

This project uses PyTorch and COCO datasets (custom dataset or LIVECell dataset) to train a Mask-RCNN model for the task of cell image segmentation.

The goal is to implement an application for loading datasets and training model for the task of instance segmentation of phase contrast microscopy sourced astrocyte images.
After training, the model can be used to segment images provided in a path in a config file.

The LIVECell dataset is sourced from 
https://github.com/sartorius-research/LIVECell

The custom dataset used is created with Label Studio.
