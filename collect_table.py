import numpy as np

import os
import os.path as path
import re

def name_split(name):
    split_list = name.split('_')
    #print(split_list)
    dataset = split_list[0]
    structure = split_list[2]
    bs = int(split_list[-8][2:])
    lr = float(split_list[-7][2:])
    wd = float(split_list[-6][2:])
    return {'dataset':dataset,'structure':structure, 'batch_size':bs, 'lr': lr, 'wd':wd}


result_dir = 'w_norm_new'

exp_dir_list = os.listdir(result_dir)

metric_list = []
test_acc_list = []
diff_list = []

param_list = []
with open('tmp_result_2.csv', 'w+') as csv:
    for exp in exp_dir_list:
        log_name = path.join(result_dir, exp,'log.txt')
        with open(log_name,'r') as f:
            lines = f.readlines()
            test_acc = float(lines[-1].split()[-1])
            tmp_list = re.split('\(|,', lines[-5])
            w_norm = float(tmp_list[-2])
            if test_acc > 1.0:
                print(exp)
                continue
            # if test_acc < 0.8:
            #     continue
            for tmp_i in range(400, len(lines)):
                if 'Testing the final model' in lines[tmp_i]:
                    train_loss = float(lines[tmp_i-1].split()[4])
                    test_loss = float(lines[tmp_i+1].split()[4])
                    diff = test_loss - train_loss
                    diff_list.append(diff)
                    break
                
        tmp_dict = name_split(exp)
        tmp_dict['test_acc'] = test_acc
        tmp_dict['metric'] = 1.0/float(lines[-2].split()[-1])
        tmp_dict['w_norm'] = w_norm
        print(tmp_dict.values())
        for k in tmp_dict:
            csv.write(str(tmp_dict[k])+',')
        csv.write('\n')
        #break
    #print(str(param_dict))
    

