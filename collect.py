from __future__ import barry_as_FLUFL
import os
import os.path as path
import numpy as np
import re
from model import resnet

result_dir = 'output/result'

exp_dir_list = os.listdir(result_dir)

metric_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}
test_acc_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}
diff_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}

metric0_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}
metric1_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}
metric2_dict = {'resnet(56,1)':[], 'resnet(32,1)':[], 'resnet(20,1)':[]}

for exp in exp_dir_list:
        print(exp)
    # try:
        if 'CIFAR10_' in exp:
            continue
        # elif 'resnet(32,1)' in exp:
        #     continue
        # elif 'resnet(20,1)' in exp:
        #     continue
        elif 'VGG' in exp:
            continue
        log_name = path.join(result_dir, exp,'log.txt')
        structure_name = exp.split('_')[2]
        assert structure_name in test_acc_dict, print(structure_name, 'is not defined')
        with open(log_name,'r') as f:
            lines = f.readlines()
            test_acc = float(lines[-1].split()[-1])
            if test_acc > 1.0:
                print(exp)
                continue
            if test_acc < 0.4:
                continue
            for tmp_i in range(400, len(lines)):
                if 'Testing the final model' in lines[tmp_i]:
                    train_loss = float(lines[tmp_i-1].split()[4])
                    test_loss = float(lines[tmp_i+1].split()[4])
                    diff = test_loss - train_loss
                    
                    diff_dict[structure_name].append(diff)
                    break

            noise_train_init_loss = []
            for tmp_i in range(len(lines)):
                if lines[tmp_i].startswith("Epoch -1 testing"):
                    noise_train_init_loss.append(float(re.findall("\d+\.\d+", lines[tmp_i])[0]))
            train_init_loss = noise_train_init_loss[0]
            noise_train_init_loss = noise_train_init_loss[1:]
            print(train_init_loss, noise_train_init_loss)

            noise_train_final_loss = []
            for tmp_i in range(len(lines)):
                if lines[tmp_i].startswith("Epoch 149 testing"):
                    noise_train_final_loss.append(float(re.findall("\d+\.\d+", lines[tmp_i])[0]))
            train_final_loss = noise_train_final_loss[0]
            noise_train_final_loss = noise_train_final_loss[1:]
            print(train_final_loss, noise_train_final_loss)

            metric0 = train_final_loss / train_init_loss / np.mean(noise_train_final_loss) * np.mean(noise_train_init_loss)
            metric1 = train_final_loss / np.mean(noise_train_final_loss)
            metric2 = train_final_loss / train_init_loss

            noise_ratio_str_list = lines[-3].replace('[',' ').replace(']',' ').replace(',',' ').split()[2:]
            noise_ratio_list = [1-float(nr) for nr in noise_ratio_str_list]
            clean_ratio = float(lines[-4].split()[-1])
            # metric = (clean_ratio)/sum(noise_ratio_list)*len(noise_ratio_list)
            # metric = clean_ratio
            metric = 1.0/float(lines[-2].split()[-1])
            
            metric_dict[structure_name].append(metric)
            test_acc_dict[structure_name].append(test_acc)
            metric0_dict[structure_name].append(metric0)
            metric1_dict[structure_name].append(metric1)
            metric2_dict[structure_name].append(metric2)
    # except:
    #     print('wrong')
    #     continue


for key in metric_dict:
    metric_dict[key] = np.array(metric_dict[key])
    test_acc_dict[key] = np.array(test_acc_dict[key])
    diff_dict[key] = np.array(diff_dict[key])
    metric0_dict[key] = np.array(metric0_dict[key])
    metric1_dict[key] = np.array(metric1_dict[key])
    metric2_dict[key] = np.array(metric2_dict[key])

plot_dict = test_acc_dict

# import matplotlib.pyplot as plt
# plt.style.use('bmh')
# fig = plt.figure(figsize=(15,12))
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.scatter(metric_dict['resnet(20,1)'], plot_dict['resnet(20,1)'], marker='o', label='ResNet20',s=220)
# plt.scatter(metric_dict['resnet(32,1)'], plot_dict['resnet(32,1)'], marker='v', label='ResNet32',s=220)
# plt.scatter(metric_dict['resnet(56,1)'], plot_dict['resnet(56,1)'], marker='^', label='ResNet56',s=220,color='black')
# plt.xlabel('$L_{\\rm{clean}}/L_{\\rm{noise}}$', fontsize=50)
# plt.ylabel('Accuracy', fontsize=50)
# plt.legend(loc='best', fontsize=45)
# # ax.grid(True, linestyle='-.')
# # plt.ylim([0.6,1])
# plt.savefig('CIFAR10_accuracy_all2')

metric_all = np.concatenate((metric_dict['resnet(20,1)'],metric_dict['resnet(32,1)'],metric_dict['resnet(56,1)']))
acc_all = np.concatenate((plot_dict['resnet(20,1)'],plot_dict['resnet(32,1)'],plot_dict['resnet(56,1)']))
metric0_all = np.concatenate((metric0_dict['resnet(20,1)'],metric0_dict['resnet(32,1)'],metric0_dict['resnet(56,1)']))
metric1_all = np.concatenate((metric1_dict['resnet(20,1)'],metric1_dict['resnet(32,1)'],metric1_dict['resnet(56,1)']))
metric2_all = np.concatenate((metric2_dict['resnet(20,1)'],metric2_dict['resnet(32,1)'],metric2_dict['resnet(56,1)']))

# print(metric_all)
print(acc_all)

print('CIFAR100:',np.corrcoef(metric_all,acc_all))
print('CIFAR100:',np.corrcoef(metric0_all,acc_all))
print('CIFAR100:',np.corrcoef(metric1_all,acc_all))
print('CIFAR100:',np.corrcoef(metric2_all,acc_all))