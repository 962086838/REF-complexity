import numpy as np
from glob import glob
import re
import pandas as pd

paths = glob("result/*")
print(len(paths))

result_df = pd.DataFrame(columns=["batch_size", "learning_rate", "dropout_rate", "test acc",
                                  "clean ratio",
                                  "L2", "L2_DIST", "PARAMS", "INVERSE_MARGIN",
                                  "LOG_PROD_OF_SPEC", "LOG_PROD_OF_SPEC_OVER_MARGIN", "LOG_SPEC_INIT_MAIN",
                                  "FRO_OVER_SPEC",
                                  "LOG_SPEC_ORIG_MAIN", "LOG_SUM_OF_SPEC_OVER_MARGIN",
                                  "LOG_SUM_OF_SPEC", "LOG_PROD_OF_FRO", "LOG_PROD_OF_FRO_OVER_MARGIN",
                                  "LOG_SUM_OF_FRO_OVER_MARGIN", "LOG_SUM_OF_FRO", "FRO_DIST",
                                  "DIST_SPEC_INIT", "PARAM_NORM", "PATH_NORM",
                                  "PATH_NORM_OVER_MARGIN", "PACBAYES_INIT", "PACBAYES_ORIG", "PACBAYES_PACBAYES_FLATNESS",
                                  "PACBAYES_PACBAYES_MAG_INIT", "PACBAYES_PACBAYES_MAG_ORIG", "PACBAYES_PACBAYES_MAG_FLATNESS",
                                  "STEP_TO_LOSS1", "STEP_TO_LOSS1point5",
                                  "extra_evaluation_metric_clean_train_loss",
                                  "extra_evaluation_metric_clean_init_loss",
                                  "extra_evaluation_metric_noisy_train_loss",
                                  "extra_evaluation_metric_noisy_init_loss",
                                  "LcleanOverLnoise", "LcleanOverLnoiseNoInit", "Lclean"
                                  ])

min_last_loss = []

for _i, each in enumerate(sorted(paths)):
    print(each)
    batch_size = re.findall("\d+", re.findall("bs\d+_", each)[0])[0]
    learning_rate = re.findall("\d+\.?\d+", re.findall("lr\d+\.?\d+_", each)[0])[0]
    dropout_rate = re.findall("\d+\.?\d+", re.findall("dropout_rate\d+\.?\d+_", each)[0])[0]
    with open(f"{each}/log.txt", 'r') as f:
        lines = f.readlines()
    for each_line in lines:
        # print(each_line)
        if each_line.startswith("clean ratio:"):
            clean_ratio = float(re.findall("-?\d+\.?\d+", each_line)[0])
            print(clean_ratio)
        if each_line.startswith("train_measures"):
            train_measures = re.findall("-?\d+\.?\d+", each_line)
            train_measures_L2 = float(train_measures[1])
            train_measures_L2_DIST = float(train_measures[3])
            train_measures_PARAMS = float(train_measures[5])
            train_measures_INVERSE_MARGIN = float(train_measures[7])
            train_measures_LOG_PROD_OF_SPEC = float(train_measures[9])
            train_measures_LOG_PROD_OF_SPEC_OVER_MARGIN = float(train_measures[11])
            train_measures_LOG_SPEC_INIT_MAIN = float(train_measures[13])
            train_measures_FRO_OVER_SPEC = float(train_measures[15])
            train_measures_LOG_SPEC_ORIG_MAIN = float(train_measures[17])
            train_measures_LOG_SUM_OF_SPEC_OVER_MARGIN = float(train_measures[19])
            train_measures_LOG_SUM_OF_SPEC = float(train_measures[21])
            train_measures_LOG_PROD_OF_FRO = float(train_measures[23])
            train_measures_LOG_PROD_OF_FRO_OVER_MARGIN = float(train_measures[25])
            train_measures_LOG_SUM_OF_FRO_OVER_MARGIN = float(train_measures[27])
            train_measures_LOG_SUM_OF_FRO = float(train_measures[29])
            train_measures_FRO_DIST = float(train_measures[31])
            train_measures_DIST_SPEC_INIT = float(train_measures[33])
            train_measures_PARAM_NORM = float(train_measures[35])
            train_measures_PATH_NORM = float(train_measures[37])
            train_measures_PATH_NORM_OVER_MARGIN = float(train_measures[39])
            train_measures_PACBAYES_INIT = float(train_measures[41])
            train_measures_PACBAYES_ORIG = float(train_measures[43])
            train_measures_PACBAYES_PACBAYES_FLATNESS = float(train_measures[45])
            train_measures_PACBAYES_PACBAYES_MAG_INIT = float(train_measures[47])
            train_measures_PACBAYES_PACBAYES_MAG_ORIG = float(train_measures[49])
            train_measures_PACBAYES_PACBAYES_MAG_FLATNESS = float(train_measures[51])
        if each_line.startswith("extra_evaluation_metric_clean_train_loss"):
            extra_evaluation_metric_clean_train_loss = float(re.findall("-?\d+\.?\d+", each_line)[0])
        if each_line.startswith("extra_evaluation_metric_clean_init_loss"):
            extra_evaluation_metric_clean_init_loss = float(re.findall("-?\d+\.?\d+", each_line)[0])
        if each_line.startswith("extra_evaluation_metric_noisy_train_loss"):
            extra_evaluation_metric_noisy_train_loss = re.findall("-?\d+\.?\d+", each_line)
            extra_evaluation_metric_noisy_train_loss = [float(_) for _ in extra_evaluation_metric_noisy_train_loss]
        if each_line.startswith("extra_evaluation_metric_noisy_init_loss"):
            extra_evaluation_metric_noisy_init_loss = re.findall("-?\d+\.?\d+", each_line)
            extra_evaluation_metric_noisy_init_loss = [float(_) for _ in extra_evaluation_metric_noisy_init_loss]
        if each_line.startswith("noisy ratio"):
            noisy_ratio = re.findall("-?\d+\.?\d+", each_line)
            noisy_ratio = [float(_) for _ in noisy_ratio]
            noisy_ratio_avr = np.mean(noisy_ratio)
        if each_line.startswith(("clean test acc")):
            test_acc = float(re.findall("-?\d+\.?\d+", each_line)[0])
    train_loss_list = []
    # for train_loss_line in lines[148:446:2]:
    for train_loss_line in lines[148:247:2]:
        print(train_loss_line)
        epoch_train_loss = re.findall("\d+[.\d]*", train_loss_line)
        train_loss_list.append(float(epoch_train_loss[1]))
    epoch_to_loss_0 = float(np.where(np.array(train_loss_list)<1.0)[0][0])
    epoch_to_loss_1 = float(np.where(np.array(train_loss_list)<1.5)[0][0])

    extra_evaluation_metric_noisy_train_loss_helper = np.mean([extra_evaluation_metric_noisy_train_loss[49], # 149
                                                               extra_evaluation_metric_noisy_train_loss[99], # 299
                                                               extra_evaluation_metric_noisy_train_loss[149], # 449
                                                               # extra_evaluation_metric_noisy_train_loss[599],
                                                               # extra_evaluation_metric_noisy_train_loss[749]
                                                               ])
    extra_evaluation_metric_noisy_init_loss_helper = np.mean([extra_evaluation_metric_noisy_init_loss[0], #
                                                              extra_evaluation_metric_noisy_init_loss[50], # 150
                                                              extra_evaluation_metric_noisy_init_loss[100], # 100
                                                              # extra_evaluation_metric_noisy_init_loss[450],
                                                              # extra_evaluation_metric_noisy_init_loss[600],
                                                              ])

    _tmp = [batch_size, learning_rate, dropout_rate, test_acc,
                         clean_ratio,
                         train_measures_L2, train_measures_L2_DIST, train_measures_PARAMS, train_measures_INVERSE_MARGIN,
                         train_measures_LOG_PROD_OF_SPEC, train_measures_LOG_PROD_OF_SPEC_OVER_MARGIN, train_measures_LOG_SPEC_INIT_MAIN,
                         train_measures_FRO_OVER_SPEC,
                         train_measures_LOG_SPEC_ORIG_MAIN, train_measures_LOG_SUM_OF_SPEC_OVER_MARGIN,
                         train_measures_LOG_SUM_OF_SPEC, train_measures_LOG_PROD_OF_FRO, train_measures_LOG_PROD_OF_FRO_OVER_MARGIN,
                         train_measures_LOG_SUM_OF_FRO_OVER_MARGIN, train_measures_LOG_SUM_OF_FRO, train_measures_FRO_DIST,
                         train_measures_DIST_SPEC_INIT, train_measures_PARAM_NORM, train_measures_PATH_NORM,
                         train_measures_PATH_NORM_OVER_MARGIN, train_measures_PACBAYES_INIT, train_measures_PACBAYES_ORIG, train_measures_PACBAYES_PACBAYES_FLATNESS,
                         train_measures_PACBAYES_PACBAYES_MAG_INIT, train_measures_PACBAYES_PACBAYES_MAG_ORIG, train_measures_PACBAYES_PACBAYES_MAG_FLATNESS,
                         epoch_to_loss_0 / float(batch_size), epoch_to_loss_1 / float(batch_size),
                         extra_evaluation_metric_clean_train_loss,
                         extra_evaluation_metric_clean_init_loss,
                         extra_evaluation_metric_noisy_train_loss_helper,
                        extra_evaluation_metric_noisy_init_loss_helper,
            extra_evaluation_metric_clean_train_loss/extra_evaluation_metric_clean_init_loss/extra_evaluation_metric_noisy_train_loss_helper*extra_evaluation_metric_noisy_init_loss_helper,
            extra_evaluation_metric_clean_train_loss/extra_evaluation_metric_noisy_train_loss_helper,
            extra_evaluation_metric_clean_train_loss/extra_evaluation_metric_clean_init_loss,
                         ]
    result_df.loc[_i] = _tmp
result_df.to_csv("result_new.csv")

result_array = np.array(result_df)
for i in range(result_array.shape[0]):
    for j in range(result_array.shape[1]):
        result_array[i, j] = float(result_array[i, j])

N = 5
mean_array = []
std_array = []
for i in range(0, result_array.shape[0], N):
    tmp_mean = np.mean(result_array[i:i+N], axis=0)
    print(tmp_mean)
    mean_array.append(tmp_mean)
    tmp_std = result_array[i:i+N]
    line_std = []
    for k in range(tmp_std.shape[1]):
        line_std.append(np.std(tmp_std[:, k]))
    line_std = np.array(line_std)
    print(line_std)
    std_array.append(line_std)
mean_array = np.array(mean_array)  # 10, 37
std_array = np.array(std_array)


results_mean_df = pd.DataFrame(columns=result_df.columns)
results_std_df = pd.DataFrame(columns=result_df.columns)
for i in range(mean_array.shape[0]):
    results_mean_df.loc[i] = mean_array[i]
for i in range(std_array.shape[0]):
    results_std_df.loc[i] = std_array[i]

results_mean_df.to_excel("result_new_mean.xlsx")
results_std_df.to_excel("result_new_std.xlsx")

