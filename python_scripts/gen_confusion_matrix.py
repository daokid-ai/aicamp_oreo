# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import traceback
from pandas_ml import ConfusionMatrix
from ai import *
from exceptions import *
from my_yolo_utils import *
from metrics_utils import *

# Section Headers from  various properties files
MODEL_EVAL_PREP_SEC = 'MODEL_EVAL_PREP'  # Section header from ai-camp-config.properties file

# properties file path
path_conf_prop = 'ai-camp-config.properties'

try:
    if not os.path.isfile(path_conf_prop):
        raise PropertiesDoesNotExistError(path_conf_prop)

    # Read Properties file for configurations
    config_dict = get_config_section(MODEL_EVAL_PREP_SEC, path_conf_prop)

    # Set Variables for Model Eval Prep configuration
    names_path = config_dict[MODEL_EVAL_PREP_SEC]['names_path']
    cfg_path = config_dict[MODEL_EVAL_PREP_SEC]['cfg_path']
    weight_path = config_dict[MODEL_EVAL_PREP_SEC]['weight_path']

    image_folder_path = config_dict[MODEL_EVAL_PREP_SEC]['img_folder_path']
    predictions_path = config_dict[MODEL_EVAL_PREP_SEC]['predictions_path']
    validated_folder_path = config_dict[MODEL_EVAL_PREP_SEC]['validated_folder_path']

    default_confidence_level = float(config_dict[MODEL_EVAL_PREP_SEC]['default_confidence_level'])
    default_threshold = float(config_dict[MODEL_EVAL_PREP_SEC]['default_threshold'])
    save_image = config_dict[MODEL_EVAL_PREP_SEC]['save_image']

    simulation_results_nm = "model_simulation.txt"

    create_file_txt(predictions_path, simulation_results_nm)
    cm_raw_figure_nm = 'cm_raw_matrix.png'
    cm_per_figure_nm = 'cm_per_matrix.png'
    # list_confidence_levels = [0.0, 1.0]
    list_confidence_levels = [0.3]
    # list_confidence_levels = [0.5]
    # list_confidence_levels = [0.3, 0.5, 0.7]
    # list_confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    # list_confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    precision_y = []
    recall_x = []

    for lcl in list_confidence_levels:
        new_validated_folder_path = os.path.join(validated_folder_path, 'confidence_{}'.format(lcl) + "/")

        if not os.path.exists(new_validated_folder_path):
            os.mkdir(new_validated_folder_path)

        predictions = yolo_pred_list(image_folder_path, new_validated_folder_path, names_path, cfg_path, weight_path,
                                     lcl, default_threshold, save_image)

        y_actual, y_predicted = get_arrys_actls_and_preds(predictions)
        # This block of code uses pandas-ml which is only compatible with pandas v0.24.0
        pd_ml_cm = get_pd_ml_cf_matrix(y_actual, y_predicted)
        # pd_ml_cm.print_stats()  # print stats

        # will need to eventually create plots
        precision_y.append(pd_ml_cm.PPV)
        recall_x.append(pd_ml_cm.NPV)

        save_pd_ml_cm_stats(predictions_path, simulation_results_nm, lcl, pd_ml_cm)  # write stats to file
        # disp_pd_ml_cf_matrix_per(pd_ml_cm)  # Command Line output of confusion matrix
        # save_cm_figure(get_fancy_pd_cf_matrix(get_pd_cf_matrix(y_actual, y_predicted)), new_validated_folder_path,
        #               cm_raw_figure_nm)  # raw confusion matrix values
        save_cm_figure(get_fancy_pd_ml_cm(pd_ml_cm), new_validated_folder_path, cm_per_figure_nm)  # with percentages

except PropertiesDoesNotExistError as pdnee:
    print("{}".format(pdnee))
except UnhandledException as ue:
    print("Some other error occurred.{}".format(ue))
except:
    traceback.print_exc()
