from os import listdir
from os.path import isfile, join

import pandas as pd

if __name__ == "__main__":
    mypath = r"D:\dev\dataset\tiff_to_png_files"
    output_excel_path = r"D:\dev\dataset\excel_columns_file_names.xlsx"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    df = pd.DataFrame(onlyfiles, columns=["file_names"])
    print(df)

    df.to_excel(output_excel_path)
