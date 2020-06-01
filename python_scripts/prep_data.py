# import necessary libraries
# import ipdb; ipdb.set_trace()
import json
import traceback
# import LabelBox objects
from labelbox import Client
from labelbox import Project
from labelbox.exceptions import ResourceNotFoundError

#import my other python classes and methods
from my_yolo_utils import *
from exceptions import *

# Section Headers from  various properties files
KEY_LOC_SEC = 'KEY_LOC'  # Section Header from ai-camp-config properties file
TRAINING_DATA_PREP_SEC = 'TRAINING_DATA_PREP'  # Section header from ai-camp-config.properties file
LABELBOX_KEY = 'LABELBOX_KEY'  # Section Header from ai-camp-keys.properties file

# Set training parameters from read ai-camp-config.properties
try:
    path_conf_prop = 'ai-camp-config.properties'
    config_dict = get_config_section(TRAINING_DATA_PREP_SEC, path_conf_prop)

    # Set path for API Key File
    path_lbl_box_api_keys = config_dict[KEY_LOC_SEC]['key_file']

    # Set Variables for Training Data Prep configuration
    my_proj_nm = config_dict[TRAINING_DATA_PREP_SEC]['project_nm']  # Set which labelbox project to find
    prep_data_dir = config_dict[TRAINING_DATA_PREP_SEC]['prep_data_dir']  # Set which directory to store prepared data
    config_dir = config_dict[TRAINING_DATA_PREP_SEC]['config_dir']  # Set directory for train.txt, test.txt
    prefix_label_nm = config_dict[TRAINING_DATA_PREP_SEC]['prefix_label_nm']  # Set prefix of prepared image name
    prep_data_on = config_dict[TRAINING_DATA_PREP_SEC]['prep_data_on']  # Download image on
    prep_label_on = config_dict[TRAINING_DATA_PREP_SEC]['prep_label_on']  # Create Label txt on
    colab_path_prep_data_dir = config_dict[TRAINING_DATA_PREP_SEC]['colab_path_prep_data_dir']  # Set colab directory
    num_classes = config_dict[TRAINING_DATA_PREP_SEC]['num_classes']  # Number of classifications

    if prep_data_on == "yes":
        prep_data_limit = int(config_dict[TRAINING_DATA_PREP_SEC]['max_image_count'])  # Set Max Number images to prep
    else:
        prep_data_limit = int(config_dict[TRAINING_DATA_PREP_SEC]['prep_data_limit'])  # Set subset of images to prep
except UnhandledException as ue:
    raise ue("Initialization of directory path and file prefixes for training data preparation. ")
    traceback.print_exc()

# Set path and read ai-camp-keys.properties
try:
    api_key_dict = get_config_section(LABELBOX_KEY, path_lbl_box_api_keys)
    LBLBX_API_KEY = api_key_dict[LABELBOX_KEY]['label_box_api_key']
except UnhandledException as ue:
    raise ue("Could not grab parameters for the LabelBox API Key.")
    traceback.print_exc()

# Fetch a Project by name
try:
    client = Client(LBLBX_API_KEY)
    #projects_x = client.get_projects(where=Project.name == "test")
    projects_x = client.get_projects(where=Project.name == my_proj_nm)
    my_proj_list = []

    for my_proj in projects_x:
        my_proj_list.append(my_proj)

    if len(my_proj_list) == 0:
        raise Exception("There are no projects assigned to user ")
    elif len(my_proj_list) > 1:
        print("There is more than one project assigned to user. Program will take the first project.")
    else:
        print(" There is one project assigned to user.")

    my_json_url = client.get_project(my_proj_list[0].uid).export_labels()
    loaded_json = json.loads(urllib.request.urlopen(my_json_url).read().decode())

except NoLabelBoxProjectsFound as nblpe:
    raise nblpe("Unknown error. Could not obtain any label box projects.")
except UnhandledException as ue:
    raise ue("Unknown error in obtaining Labelbox project data set ")
    traceback.print_exc()

sample_count = 0  # file_name numbering
skipped_count = 0  # number of files with no labels
create_file_txt(config_dir, "train.txt")  # First time running overwrite previous train.txt or create new one

for json_row in loaded_json:
    try:
        if json_row.get("Label") == "Skip":
            skipped_count = skipped_count + 1
        else:
            sample_count = sample_count + 1
            part_file_nm = set_file_prename(sample_count, prefix_label_nm)  # partial file name

            labeled_data_url = json_row.get("Labeled Data")  # retrieve image url
            lbl_bx_coordinates = (json_row.get("Label").get("Oreo"))[0].get("geometry")

            if sample_count <= prep_data_limit:

                download_image(labeled_data_url, prep_data_dir, part_file_nm)
                image_dimensions = get_image_size(prep_data_dir, part_file_nm)
                add_img_to_train_txt(config_dir, colab_path_prep_data_dir, part_file_nm)

                prepared_data = MyYoloFormat()
                prepared_data.set_yolo_format_vals(num_classes, image_dimensions, lbl_bx_coordinates)
                create_label_txt(prep_data_dir, prepared_data, part_file_nm)

    except UnhandledException as ue:
        raise ue("One of the methods in the try statement had an issue with file {}".format(sample_count)
                 + " proceeding to next file.")
        pass

print("{}".format(sample_count) + " images prepared. {}".format(skipped_count) + " images skipped.")
print("\n=================== Training Data has been prepared. ====================")
