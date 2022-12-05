import nibabel
import data_manipulation as dm
import data_visualization as dv
import numpy as np


def main():

    # dm.list_files()

    data_dict = dm.load_dict("img_and_annot.json")

    labels = data_dict["r_annot"]

    label_img_1 = dm.read_img(labels[15])
    label_img_2 = dm.read_img(labels[4])

    labels_1 = np.unique(label_img_1.get_fdata())
    labels_2 = np.unique(label_img_2.get_fdata())

    for i in range(len(labels_1)):
        print(f"{labels_1[i]}  {labels_2[i]}")

    print(len(labels_1))


if __name__ == "__main__":
    main()