# import necessary libraries
# import ipdb; ipdb.set_trace()
import os
import requests
import configparser
import urllib.request
import numpy as np
from PIL import Image
from exceptions import *
import cv2

################################ START OF CLASS DEFINITIONS ###################################


class MyYoloFormat:
    def __init__(self):
        self.name = ""
        self.classes = 0
        self.img_height = 1
        self.img_width = 1
        self.center_x_fraction = 0.1
        self.center_y_fraction = 0.1
        self.width_box_fraction = 0.1
        self.height_box_fraction = 0.1

    def set_yolo_format_vals(self, classes, img_dimensions, boundingbox):
        """
        cv2 dimensions are in this order height, width, & channels
        Image dimensions are in this order width, & height
        """
        self.name = "MyYoloFormatClass"
        self.classes = int(classes) - 1
        img_height, img_width, _ = img_dimensions
        x_coords = []
        y_coords = []

        for coords in boundingbox:
            x_coords.append(coords.get('x'))
            y_coords.append(coords.get('y'))

        center_x_box = (np.max(x_coords) + np.min(x_coords)) / 2.0
        center_y_box = (np.max(y_coords) + np.min(y_coords)) / 2.0
        self.center_x_fraction = center_x_box / img_width
        self.center_y_fraction = center_y_box / img_height
        box_width = np.max(x_coords) - np.min(x_coords)
        box_height = np.max(y_coords) - np.min(y_coords)
        self.width_box_fraction = box_width / img_width
        self.height_box_fraction = box_height / img_height

    def get_yolo_format_val(self):
        return [self.classes, self.center_x_fraction, self.center_y_fraction,
                self.width_box_fraction, self.height_box_fraction]

################################ END OF CLASS DEFINITIONS #######################################

################################ START OF METHODS DEFINITIONS ###################################


# method to read section header of properties file.
def get_config_section(section_header, path_conf_prop):
    try:
        cfg_session = configparser.ConfigParser()
        cfg_session.read(path_conf_prop)
        if not hasattr(get_config_section, section_header):
            get_config_section.section_dict = dict()
            for section in cfg_session.sections():
                get_config_section.section_dict[section] = dict(cfg_session.items(section))
        return get_config_section.section_dict
    except Exception as e:
        raise e("Could Not get parameters from properties files.")


# method for creating a new train.txt file
def create_file_txt(dest_dir,file_name):
    file_path_nm = os.path.join(dest_dir, file_name)
    try:
        with open(file_path_nm, "w") as train_txt:
            print("Created {}".format(file_path_nm))
    except Exception as e:
        raise e("Error while creating {}".format(file_path_nm))


# method for setting downloaded image and label.txt file name
def set_file_prename(sample_count, prefix_label_nm):
    try:
        trn_dat_num = "{}".format(sample_count).zfill(4)  # file number suffix
        part_file_nm = prefix_label_nm + trn_dat_num  # partial file name
        return part_file_nm
    except UnhandledException as ue:
        raise ue("Could not set the pre file name:")


# method to download image from url
def download_image(url, prep_data_path, part_file_name):
    complete_file_name = part_file_name + ".jpg"
    try:
        response = requests.get(url)
        if not response:
            raise Exception(" there is no response from url.")

        with open(os.path.join(prep_data_path, complete_file_name),'wb') as img_file:
            img_file.write(response.content)
            print("Saved image as {}".format(os.path.join(prep_data_path, complete_file_name)))
    except Exception as e:
        raise e("Error when downloading image to {}".format(prep_data_path))


# method to return the image size
def get_image_size(prep_data_path, part_file_name):
    complete_file_name = part_file_name + ".jpg"
    try:
        image = cv2.imread(os.path.join(prep_data_path, complete_file_name))
        # print("\tImage dimensions of {}".format(complete_file_name) + " are {}".format(image.shape))
        return image.shape
    except Exception as e:
        raise e("Could not get image size from {}".format(complete_file_name))


# method to adding image path and name to train.txt
def add_img_to_train_txt(dest_dir, colab_path_prep_data_dir, part_file_name):
    complete_file_name = part_file_name + ".jpg"
    train_txt_path = dest_dir + "train.txt"
    try:
        with open(train_txt_path, "a") as train_txt:
            train_txt.write("\n{}".format(os.path.join(colab_path_prep_data_dir, complete_file_name)))
            print("\tAdded {}".format(complete_file_name) + " to {}".format(train_txt_path))
    except Exception as e:
        raise e("Could not add img path to {}".format(train_txt_path))


# method for creating label text file
def create_label_txt(prep_data_path, label_data, part_file_name):
    complete_file_name = part_file_name + ".txt"
    try:
        with open(os.path.join(prep_data_path, complete_file_name), "w") as lbtf:
            lbtf.write("{}".format(label_data.classes)
                       + " {}".format(label_data.center_x_fraction) + " {}".format(label_data.center_y_fraction)
                       + " {}".format(label_data.width_box_fraction) + " {}".format(label_data.height_box_fraction))
            print("\tCreated: {}".format(complete_file_name) +
                  " with following values {}".format(label_data.get_yolo_format_val()))
    except Exception as e:
        raise e("Could not create label file here {}".format(prep_data_path))


# method for determining digits places for renaming file
def set_file_numbering(len_file_list):
    temp_num = len_file_list
    digit_places = 0
    while temp_num > 0:
        temp_num = temp_num // 10
        digit_places = digit_places + 1
    return digit_places


############################## END METHOD DEFINITIONS ############################################