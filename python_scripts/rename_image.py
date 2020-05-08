# import necessary libraries
import os
import imghdr
import shutil
import traceback
from my_yolo_utils import *
from exceptions import *

LABEL_DATA_PREP_sec ='LABEL_DATA_PREP'

try:
    path_conf_prop = 'ai-camp-config.properties'
    config_dict = get_config_section(LABEL_DATA_PREP_sec, path_conf_prop)
    src_img_dir = config_dict[LABEL_DATA_PREP_sec]['src_img_dir']
    dest_img_dir = config_dict[LABEL_DATA_PREP_sec]['dest_img_dir']
    prefix_img_nm = config_dict[LABEL_DATA_PREP_sec]['prefix_img_nm']

    if not os.path.exists(src_img_dir):
        raise PathDoesNotExistError(" {}".format(src_img_dir))
    elif not os.path.exists(dest_img_dir):
        raise PathDoesNotExistError(" {}".format(dest_img_dir))
    else:
        file_list = os.listdir(src_img_dir)
        len_file_list = len(file_list)

        if len_file_list == 0:
            raise EmptyDirectoryError()

        jpg_counter = 0
        non_jpg_counter = 0
        digit_places = set_file_numbering(len_file_list)
        print("The directory " + src_img_dir + " has {}".format(len_file_list) + " files.")
        print("Images will be renamed with numbering with {}".format(digit_places) + " digit places.\n")

        for file_name in file_list:
            eval_file = src_img_dir + file_name
            if imghdr.what(eval_file) == 'jpeg':

                file_num = str(jpg_counter + 1).zfill(digit_places)
                new_file_name = prefix_img_nm + str(file_num) + ".jpg"
                pre_processed_file = dest_img_dir + file_name
                processed_file = dest_img_dir + new_file_name
                print("Renaming {}".format(pre_processed_file) +
                      " ===========> {}".format(processed_file))  # Debugging file naming

                shutil.copy(eval_file, dest_img_dir)  # copy the file to destination dir
                os.rename(pre_processed_file, processed_file)
                jpg_counter = jpg_counter + 1
            else:
                non_jpg_counter = non_jpg_counter + 1

        print("Renamed {}".format(jpg_counter) + " Jpeg images and ignored {}".format(
            non_jpg_counter) + " non Jpeg images.")
        # print(os.listdir(dest_img_dir))  # List Files within destination directory

except PathDoesNotExistError as pdnee:
    print("{}".format(pdnee))
except EmptyDirectoryError as ede:
    print("{}".format(ede))
except UnhandledException as ue:
    print("Some other error occurred.{}".format(ue))
    traceback.print_exc()
except Exception as e:
    print("Error that UnHandled Exception did not handle: {}".format(e))
