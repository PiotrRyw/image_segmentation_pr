from copy import deepcopy
from os import listdir
from os.path import isfile, join
from typing import Dict

import pandas as p
import json
import re

def load_coco_dataset_from_file(file_name: str):
    with open(file_name, "r") as file:
        json_data_object = json.load(file)
    return json_data_object

def delete_image_filename_prefix(json_data_obj,
                                 keyword_for_images="",
                                 keyword_for_image_path="",
                                 prefix="",
                                 splitting_char="",
                                 ):

    for obj in json_data_obj[keyword_for_images]:
        obj[keyword_for_image_path] = obj[keyword_for_image_path].strip(prefix).split(splitting_char)[1]

    return json_data_obj

def change_frame_number(json_data_obj,
                        keyword_for_images="",
                        keyword_for_image_path="",
                        prefix="",
                        splitting_char="",
                        ):
    for obj in json_data_obj[keyword_for_images]:

        name: str = obj[keyword_for_image_path]  # type: ignore

        result = re.search(r"_\d*.png", name)
        # print(result, end=" ")
        if result is not None:
            num = int(name[result.start()+1:result.end()-4])
            new_name = name[0:result.span()[0]] + f"_{num:0>3}.png" + name[result.span()[1]:0]
        else:
            new_name = name
        obj[keyword_for_image_path] = new_name

    return json_data_obj

def change_file_extension(json_data_obj,
                          keyword_for_images="",
                          keyword_for_image_path="",
                          old_extension = "",
                          new_extension = "",
                          ):
    for obj in json_data_obj[keyword_for_images]:
        obj[keyword_for_image_path] = obj[keyword_for_image_path].replace(old_extension, new_extension)

    return json_data_obj


def add_original_filename(json_data_obj,
                          keyword_for_images="",
                          keyword_for_image_path="",
                          original_filename_keyword = "",
                          ):
    for obj in json_data_obj[keyword_for_images]:
        obj[original_filename_keyword] = obj[keyword_for_image_path]

    return json_data_obj

def change_categories_to_defined(json_data_obj,
                                 categories_json_obj,
                                 categories_keyword="",
                                 ):

    json_data_obj[categories_keyword] = categories_json_obj
    return json_data_obj

def find_max_image_id(json_obj):
    max_id = 0
    for img in json_obj["images"]:
        if img["id"] > max_id:
            max_id = img["id"]
    return max_id

def find_max_ann_id(json_data_obj):
    max_id = 0
    for ann in json_data_obj["annotations"]:
        if ann["id"] > max_id:
            max_id = ann["id"]
    return max_id


def drop_nonvalid_category(json_data_obj: Dict, valid):
    for ann in json_data_obj["annotations"]:
        if ann["category_id"] != valid:
            print(ann)
            json_data_obj["annotations"].remove(ann)
    return json_data_obj

def combine_multiple_datasets(file_list):
    combined = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": []
    }
    min_img_id = 0
    min_ann_id = 0

    for json_file in file_list:
        print("-------------------------------------------")
        print(json_file)
        json_data_obj = convert_single_json_dataset(json_file)
        result_json = deepcopy(json_data_obj)
        for img in result_json["images"]:
            img["id"] += min_img_id
        for ann in result_json["annotations"]:
            ann["id"] += min_ann_id
            ann["image_id"] += min_img_id

        min_img_id = find_max_image_id(json_data_obj)
        min_ann_id = find_max_ann_id(json_data_obj)
        combined["images"].extend(result_json["images"])
        combined["annotations"].extend(result_json["annotations"])

    categories = [
        {
            "supercategory": "cell",
            "id": 1,
            "name": "cell"
        }
    ]

    combined = change_categories_to_defined(combined, categories, "categories")

    return combined

def convert_single_json_dataset(input_path: str) -> Dict:
    json_data_object = load_coco_dataset_from_file(input_path)

    json_data_object = delete_image_filename_prefix(json_data_object,
                                                    keyword_for_images="images",
                                                    keyword_for_image_path= "file_name",
                                                    prefix=r"..\\..\\label-studio\\label-studio\\media\\upload\\4\\",
                                                    splitting_char = "-",
                                                    )
    json_data_object = change_frame_number(json_data_object,
                                           keyword_for_images="images",
                                           keyword_for_image_path="file_name",
                                           )

    json_data_object = add_original_filename(json_data_object,
                                             keyword_for_images="images",
                                             keyword_for_image_path="file_name",
                                             original_filename_keyword="original_filename"
                                             )

    json_data_object = change_file_extension(json_data_object,
                                             keyword_for_images="images",
                                             keyword_for_image_path= "file_name",
                                             old_extension="png",
                                             new_extension="tif"
                                             )

    json_data_object = drop_nonvalid_category(json_data_object, 1)

    return json_data_object


if __name__ == "__main__":
    # label_studio_coco_path = r"D:\dev\astrocyte-dataset\21.05\result.json"
    # label_studio_coco_path = r"D:\dev\astrocyte-dataset\label-studio-natalia\result.json"
    # output_file_path = r"D:\dev\astrocyte-dataset\label-studio-natalia\dataset_annotation.json"
    # label_studio_coco_path = r"D:\dev\astrocyte-dataset\nowe_od_natalii\result.json"
    # output_file_path = r"D:\dev\astrocyte-dataset\nowe_od_natalii\dataset_annotation.json"
    # input_files = [r"D:\dev\astrocyte-dataset\test_set\result.json"]
    # output_file_path = r"D:\dev\astrocyte-dataset\test_set\test_annotation.json"
    input_files = [r"D:\dev\astrocyte-dataset\label_studio_28_08\result.json", r"D:\dev\astrocyte-dataset\nowe_od_natalii\result.json"]
    output_file_path = r"D:\dev\astrocyte-dataset\nowe_od_natalii\dataset_annotation.json"

    json_output = combine_multiple_datasets(input_files)

    num = 0
    for img in json_output["images"]:
        num+=1
    print(num)
    print(json_output["images"])

    with open(output_file_path, "w") as file:
        json.dump(json_output, file)
