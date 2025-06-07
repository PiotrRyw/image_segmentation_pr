import os
from os import listdir
from os.path import isfile, join
from shutil import copy2

if __name__ == "__main__":
    mypath = r"D:\dev\astrocyte-dataset\original_tiff_files"
    os.chdir(mypath)

    onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

    output_directory = r"D:\dev\astrocyte-dataset\Astrocytes_dataset_2025\images\all_images"

    cohort_name = onlyfiles[0][ 0 : onlyfiles[0].find("_pos_") ]
    for original_filename in onlyfiles:
        print(original_filename, end="\t-> ")
        output_name = cohort_name + original_filename[original_filename.find("_pos_") + 5: original_filename.find(
            "_pos_") + 8] + "_" + f'{original_filename[original_filename.find("frame_") + 6: original_filename.find("of")]:0>3}' + ".tif"
        print(join(output_directory, output_name))

        copy2(original_filename, join(output_directory, output_name))
