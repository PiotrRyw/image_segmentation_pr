from pathlib import Path

import pandas as pd

from scripts.img_seg_dataset import COCOLIVECellDataset, DatasetUtils
from scripts.project_utils import train_tfms, OrientationCorrection


def open_all_images_and_annotations_from_dataset():
    #
    # image_dict, annotation_df, image_keys, val_keys, class_to_idx = DatasetUtils.create_image_and_annotation_dict(
    #     Path(r"D:\dev\dataset\21.05\images\\"),
    #     r"D:\dev\dataset\21.05\result.json",
    #     ["cell", "background"],
    #     1,
    # )
    pd.options.display.max_columns = None
    pd.options.display.width = 0

    annotation_file_path =  r"D:\dev\dataset\21.05\result.json"
    annotation_file_path = r"D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/0_train2percent.json"

    annotation_file_df = pd.read_json(annotation_file_path, orient='index').transpose()
    categories_df = annotation_file_df['categories'].dropna().apply(pd.Series)
    categories_df.set_index('id', inplace=True)
    # pd.options.display.max_columns = None
    # pd.options.display.width = 0
    # Extract and transform the 'images' section of the data
    # This DataFrame contains image details like file name, height, width, and image ID
    images_df = annotation_file_df['images'].to_frame()['images'].apply(pd.Series)[
        ['file_name', 'height', 'width', 'id']]

    print(f"Extracted image file data")

    # Extract and transform the 'annotations' section of the data
    # This DataFrame contains annotation details like image ID, segmentation points, bounding box, and category ID
    annotations_df = annotation_file_df['annotations'].to_frame()['annotations'].apply(pd.Series)[
        ['image_id', 'segmentation', 'bbox', 'category_id']]

    # Map 'category_id' in annotations DataFrame to category name using categories DataFrame
    annotations_df['label'] = annotations_df['category_id'].apply(lambda x: categories_df.loc[x]['name'])

    # Merge annotations DataFrame with images DataFrame on their image ID
    annotation_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id')

    print("Images dataframe:")
    print(images_df.head())
    print("annotations_df dataframe:")
    print(annotations_df.head())
    print("annotation_df dataframe:")
    print(annotation_df.head())




    # image_dict, annotation_df, image_keys, val_keys, class_to_idx = DatasetUtils.create_image_and_annotation_dict(
    #     Path(r"D:/dev/livecell-dataset/LIVECell_dataset_2021/images/all_images/"),
    #     r"D:/dev/livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/0_train2percent.json",
    #     ["cell", "background"],
    #     0.5,
    # )

    orientation_corr = OrientationCorrection()
    orientation_corr.define_height_width(
        height=520,
        width=696,
    )

    # print(image_dict, "\n", annotation_df, image_keys, val_keys, class_to_idx)
    #
    # dataset = COCOLIVECellDataset(image_keys,
    #                               annotation_df,
    #                               image_dict,
    #                               class_to_idx,
    #                               orientation_corr,
    #                               train_tfms,
    #                               )


def create_labelled_image_from_annotation():
    pass


if __name__ == '__main__':
    open_all_images_and_annotations_from_dataset()
