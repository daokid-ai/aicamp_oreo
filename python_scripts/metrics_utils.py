# import necessary libraries
# import ipdb; ipdb.set_trace()
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from exceptions import *

################################ START OF CLASS DEFINITIONS ###################################


# method for saving predictions to file
def save_predictions(predictions, predictions_path):

    if not predictions_path or not predictions_path:
        raise Exception("One of the arguments to save_predictions is not set up.")

    file_path_nm = os.path.join(predictions_path, "model_predictions.txt")
    try:
        with open(file_path_nm, "w") as model_predictions:
            model_predictions.write("{}".format(predictions))
            print("Created {}".format(file_path_nm))
    except Exception as e:
        raise e("Error while creating {}".format(file_path_nm))


# generating 2 arrays of actual and predicted values
def get_arrys_actls_and_preds(predictions):

        if not predictions:
            raise Exception('No Predictions')

        y_actual = []
        y_predicted = []

        for pred in predictions:
            image_path = pred.get('image_path')
            if 'Oreo' in image_path:
                y_actual.append(1)
            else:
                y_actual.append(0)

            class_ids = pred.get('class_ids')
            if len(class_ids) == 0:
                y_predicted.append(0)
            else:
                y_predicted.append(1)
                # print("ClassID is : {}".format(class_ids))

        return y_actual, y_predicted


# create simple formatted confusion matrix using pandas data frame
def get_pd_cf_matrix(y_actual, y_predicted):
    if not y_actual or not y_predicted:
        raise Exception('Actuals or Predictions are missing.')

    data = {'y_Actual': y_actual,
            'y_Predicted': y_predicted
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    pd_cf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)
    return pd_cf_matrix


# convert simple to  fancy formatted confusion matrix using seaborn and pandas data frame
def get_fancy_pd_cf_matrix(pd_cf_matrix):
    svm = sn.heatmap(pd_cf_matrix, annot=True)
    return svm.get_figure()


# get fancy confusion matrix with percentages using seaborn
def get_fancy_pd_ml_cm(cf_matrix):
    cf_2darry = [[cf_matrix.TN, cf_matrix.FP ],[cf_matrix.FN, cf_matrix.TP]]

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(cf_matrix.TN),"{0:0.0f}".format(cf_matrix.FP),
                    "{0:0.0f}".format(cf_matrix.FN), "{0:0.0f}".format(cf_matrix.TP)]
    group_percentages = ["{0:.2%}".format(cf_matrix.TN/cf_matrix.N), "{0:.2%}".format(cf_matrix.FP/cf_matrix.N),
                         "{0:.2%}".format(cf_matrix.FN / cf_matrix.P), "{0:.2%}".format(cf_matrix.TP/cf_matrix.P)]
    x_labels = ['No Oreo','Oreo' ]
    y_labels = ['No Oreo','Oreo' ]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    svm_plus = sn.heatmap(cf_2darry, annot=labels, fmt='', cmap='GnBu', xticklabels=x_labels, yticklabels=y_labels)
    svm_plus.set(title="Confusion Matrix",
            xlabel="Predicted",
            ylabel="Actual",)
    return svm_plus.get_figure()
    # plt.show()


# save figure of confusion matrix to disk
def save_cm_figure(figure, dest_path, figure_nm):
    file_path_nm = os.path.join(dest_path, figure_nm)
    figure.savefig(file_path_nm, dpi=400)


# generating Confusion Matrix stats from pandas_ml Confusion Matrix
def get_pd_ml_cf_matrix(y_actual, y_predicted):

    data = {'y_Actual':    y_actual,
            'y_Predicted': y_predicted
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    pd_ml_cf_matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])

    return pd_ml_cf_matrix

# display confusion matrix in percentages
def disp_pd_ml_cf_matrix_per(pd_ml_cf_matrix):
    if not pd_ml_cf_matrix:
        raise Exception('No Confusion Matrix Percentages')

    tpp = round(pd_ml_cf_matrix.TP / pd_ml_cf_matrix.P, 2)
    fnp = round(pd_ml_cf_matrix.FN / pd_ml_cf_matrix.P, 2)
    fpp = round(pd_ml_cf_matrix.FP / pd_ml_cf_matrix.N, 2)
    tnp = round(pd_ml_cf_matrix.TN / pd_ml_cf_matrix.N, 2)

    print("  ---------Predicted Labels------------ ")
    print(" |- True Positive %-|-False Negative %-|")
    print("T|                                     |")
    print("R|       {:.2f}".format(tpp) + "       |       {:.2f}".format(fnp) + "       |")
    print("U|                                     |")
    print("E|-False Positive %-|- True Negative %-|")
    print(" |                                     |")
    print("L|       {:.2f}".format(fpp) + "       |       {:.2f}".format(tnp) + "       |")
    print("B|                                     |")
    print("L|------------------|------------------|")


# method for saving metrics against confidence thresholds..
def save_pd_ml_cm_stats(predictions_path, file_name, conf_threshold, pd_ml_cm):

    if not predictions_path or not file_name or not pd_ml_cm:
        raise Exception("One of the arguments to save stats is not set up.")

    file_path_nm = os.path.join(predictions_path, file_name)
    with open(file_path_nm, "a") as simul_results:
        simul_results.write("\n\nConfidence Threshold Level: {}".format(conf_threshold) +
                            "\n# of TP : {}".format(pd_ml_cm.TP) +
                            "\n# of TN: {}".format(pd_ml_cm.TN) +
                            "\n# of FP : {}".format(pd_ml_cm.FP) +
                            "\n# of FN: {}".format(pd_ml_cm.FN) +
                            "\nPrecision : {}".format(pd_ml_cm.PPV) +
                            "\nRecall : {}".format(pd_ml_cm.NPV) +
                            "\nArea(PXR) :{}".format(pd_ml_cm.PPV*pd_ml_cm.NPV) +
                            "\nAccuracy : {}".format(pd_ml_cm.ACC) +
                            "\nF1 Score : {}".format(pd_ml_cm.F1_score))


############################## END METHOD DEFINITIONS ############################################