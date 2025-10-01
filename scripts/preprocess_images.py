# Code from Sartorius Corporate Research on GitHub
import os.path

import cv2  # type: ignore
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt  # type: ignore

# this function is designed to adapt images acquired with other light microscopy modalities
# in order to enable inference with LIVECell trained models

# input_image = uint8 numpy array
# magnification_resample_factor = downsample factor needed to make image effective 10X
#   for 40x images, use 0.25
#   for 20x images, use 0.5
#   for 10x images, use 1

def preprocess(input_image, magnification_downsample_factor=1.0):
    # internal variables
    #   median_radius_raw = used in the background illumination pattern estimation.
    #       this radius should be larger than the radius of a single cell
    #   target_median = 128 -- LIVECell phase contrast images all center around a 128 intensity
    median_radius_raw = 75
    target_median = 128.0

    # large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw * magnification_downsample_factor)
    if median_radius % 2 == 0:
        median_radius = median_radius + 1

    # scale so mean median image intensity is 128
    input_median = np.median(input_image)
    intensity_scale = target_median / input_median
    output_image = input_image.astype('float') * intensity_scale

    # define dimensions of downsampled image
    dims = input_image.shape
    y = int(dims[0] * magnification_downsample_factor)
    x = int(dims[1] * magnification_downsample_factor)

    # apply resizing image to account for different magnifications
    output_image = cv2.resize(output_image, (x, y), interpolation=cv2.INTER_AREA)

    # clip here to regular 0-255 range to avoid any odd median filter results
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0

    # estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    output_image = output_image.astype('float') / background.astype('float') * target_median

    # clipping for zernike phase halo artifacts
    output_image[output_image > 180] = 180
    output_image[output_image < 70] = 70
    output_image = output_image.astype('uint8')

    return output_image


# Code written by PR:
from PIL import Image

def script_change_tif_to_png():
    mypath = r"D:\dev\dataset\original_tiff_files"
    out_put_dir = r"D:\dev\dataset\tiff_to_png_files"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)

    cohort_name = onlyfiles[0][ 0 : onlyfiles[0].find("_pos_") ]

    for filepath in onlyfiles:

        cohort_name = filepath[0: filepath.find("_pos_")]
        output_name = cohort_name + filepath[filepath.find("_pos_")+5 : filepath.find("_pos_")+8 ] + "_" + f'{filepath[filepath.find("frame_")+6 : filepath.find("of")]:0>3}' + ".png"

        output_file_path = out_put_dir + "\\" + output_name


        filepath = mypath + "\\" + filepath

        input_image = Image.open(filepath).convert('RGB')
        output_image = preprocess(np.asarray(input_image))
        proccessed_image = Image.fromarray(output_image)
        proccessed_image.save(output_file_path)
        print(f"Saved to {output_file_path}")

def script_batch_preprocess():
    image_dir = r"D:\dev\astrocyte-dataset\Astrocytes_dataset_2025\images\all_images"
    onlyfiles = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    for filepath in onlyfiles:
        filepath = os.path.join(image_dir, filepath)
        input_image = Image.open(filepath).convert('RGB')
        output_image = preprocess(np.asarray(input_image))
        proccessed_image = Image.fromarray(output_image)
        proccessed_image.save(filepath)
        print(f"Saved to {filepath}")

def script_batch_correct_livecell():
    image_dir = r"D:\dev\livecell-dataset\LIVECell_dataset_2021\images\all_images"
    new_image_dir = r"D:\dev\livecell-dataset\LIVECell_dataset_2021\images\altered_images"
    onlyfiles = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    threshold = 127 + 14

    for file_path in onlyfiles:
        input_path = image_dir + "\\" + file_path
        output_path = new_image_dir + "\\" + file_path
        print(input_path, output_path)

        org_img = cv2.imread(input_path)
        img = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)

        high_value_mask = img[:, :, 2] > threshold
        img[:, :, 2][high_value_mask] = 129
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    # script_batch_preprocess()

    script_batch_correct_livecell()
