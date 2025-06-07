from os import listdir
from os.path import isfile, join

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

        name: str = obj[keyword_for_image_path]

        result = re.search(r"_\d*.png", name)
        print(result, end=" ")
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

if __name__ == "__main__":
    label_studio_coco_path = r"D:\dev\astrocyte-dataset\21.05\result.json"

    output_file_path = r"D:\dev\astrocyte-dataset\21.05\dataset_annotation.json"

    json_data_object = load_coco_dataset_from_file(label_studio_coco_path)

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

    print(json_data_object["images"])

    categories = [
        {
            "supercategory": "cell",
            "id": 1,
            "name": "cell"
        }
    ]

    json_data_object = change_categories_to_defined(json_data_object, categories, "categories")

    print(json_data_object["categories"])

    with open(output_file_path, "w") as file:
        json.dump(json_data_object, file)
