import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

import random
import scipy.io
import codecs
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Models.SiameseModel import Siamese
from Datasets.Physionet.NoisyDataset import PairsDataset

# ecg1 = np.transpose(scipy.io.loadmat('Data\PTB-XL\Train\Clean\\00010.mat')['ECG'])
# ecg2 = np.transpose(scipy.io.loadmat('Data\PTB-XL\Train\Clean\\00020.mat')['ECG'])
# ecg3 = np.transpose(scipy.io.loadmat('Data\PTB-XL\Train\Clean\\00030.mat')['ECG'])
# ecg4 = np.transpose(scipy.io.loadmat('Data\PTB-XL\Train\Clean\\00050.mat')['ECG'])
ecg5 = np.transpose(scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\A0149.mat')['ECG'])[0][0][2]
# ecg1 = scipy.io.loadmat('Data\PTB-XL\Train\Filtered\\00010.mat')['ECG']
# ecg2 = scipy.io.loadmat('Data\PTB-XL\Train\Filtered\\00020.mat')['ECG']
# ecg3 = scipy.io.loadmat('Data\PTB-XL\Train\Filtered\\00030.mat')['ECG']
# ecg4 = scipy.io.loadmat('Data\PTB-XL\Train\Filtered\\00050.mat')['ECG']
# ecg5 = scipy.io.loadmat('Data\PTB-XL\Train\Filtered\\00060.mat')['ECG']
# ecg1 = scipy.io.loadmat('Data\PTB-XL\Train\Prepared_Noisy\\20_clean.mat')['ECG']
# ecg2 = scipy.io.loadmat('Data\PTB-XL\Train\Prepared_Noisy\\40_clean.mat')['ECG']
# ecg3 = scipy.io.loadmat('Data\PTB-XL\Train\Prepared_Noisy\\60_clean.mat')['ECG']
# ecg4 = scipy.io.loadmat('Data\PTB-XL\Train\Prepared_Noisy\\100_clean.mat')['ECG']
# ecg5 = scipy.io.loadmat('Data\PTB-XL\Train\Prepared_Noisy\\120_clean.mat')['ECG']
# ecg = np.concatenate((ecg1[0], ecg2[0]))
# ecg = np.concatenate((ecg, ecg3[0]))
# ecg = np.concatenate((ecg, ecg4[0]))
# ecg = np.concatenate((ecg, ecg5[0]))
plt.plot(ecg5[0])
plt.show()

# ds_noisy = PairsDataset(WITH_ROLL=True)
# pair, label = ds_noisy.__getitem__(12) # Different but looks same: 8, 10

# model = Siamese()
# model.load_state_dict(torch.load('nets\SCNN.pth'))

# in1 = pair[0][None, :, :]
# in2 = pair[1][None, :, :]

# model.train(False)
# print('predicted: ', model(in1, in2).item())
# print('true: ', label.item())

# fig, axs = plt.subplots(2)
# axs[0].plot(pair[0][0])
# axs[1].plot(pair[1][0])
# plt.show()


# chinese_dtst_reference = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')
# ecg = scipy.io.loadmat('ChineseDataset\TrainingSet1\A1445.mat')['ECG'][0][0][2]
# L = len(ecg[0])
# x = np.linspace(0, L, L)
# A = np.random.uniform(0.05, 0.4)
# T = 2 * L
# noise = np.concatenate((np.zeros((1, L)), np.zeros((1, L))), axis=0)
# wander = []
# PHI = np.random.uniform(0, 2 * math.pi)
# for i in x:
#     wander.append(A * np.cos(2 * math.pi * (i/T) + PHI))
# noise = np.sum([noise, wander], axis=0)
# print(noise)
# plt.plot(ecg[0] + noise[0])
# plt.show()

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ds = PairsDataset(DEVICE, fill_with_type='mean')

# for i in range(ds.__len__()):
#     ecg = ds.__getitem__(1)
#     plt.plot(ecg[6][0][0])
#     plt.show()
#     break

# print(mat)

# import matplotlib.pyplot as plt
# from PreprocessingFilters import filter1
# import math

# struct1 = np.ones((mat['ECG'][0][0][2].shape[0], 6)) / 5
# struct2 = np.ones((mat['ECG'][0][0][2].shape[0], 45)) / 5
# data1 = filter1(mat['ECG'][0][0][2], struct1, struct2)[:, 100:-100]

# mean = []
# if (15000 - data1.shape[1]) > 0:
#     for i in range(12):
#         mean.append(np.full(15000 - data1.shape[1], np.mean(data1[i])))
#     print(data1.shape)
#     ekg = np.column_stack([data1, mean])
# else:
#     ekg = data1[:, :15000]
# print(ekg.shape)

# _, axs = plt.subplots(2)
# axs[0].plot(ekg[1, :])

# means = [5.6671288368373204e-08, -5.672094472486019e-08, -1.1381123812568519e-07,
#                  -7.73628575187182e-10, 8.544064961353723e-08, -8.466800578420468e-08,
#                  -5.644898281745803e-08, -3.3201897366838757e-07, -6.639807663731727e-08,
#                  -1.9771499946997733e-08, -3.3253429074075554e-08, 1.487236435452322e-07]
# stds = [0.2305271687030844, 0.24780370485706876, 0.23155043161905942,
#         0.22074153961314927, 0.20708526280758174, 0.22153766144293813,
#         0.353942949952694, 0.3942397032518631, 0.4228515959530688,
#         0.436324876078121, 0.47316252072611537, 0.5328047065188085]

# for i in range(12):
#     ekg[i, :] = (ekg[i, :] - means[i]) / stds[i]
# axs[1].plot(ekg[1, :])

# plt.show()



# import math
# stats = {}
# dataframe = chinese_dtst_reference.loc[chinese_dtst_reference['Recording'] <= 'A2000'].reset_index(drop=True)
# max_shape = 0
# min_shape = math.inf
# for i in range(len(dataframe.index)):
#     current_mat = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + dataframe['Recording'][i] + '.mat')['ECG'][0][0][2]
#     current_shape = current_mat.shape[1]

#     if current_shape not in stats.keys(): stats[current_shape] = 1
#     else: stats[current_shape] += 1

#     if max_shape < current_shape: max_shape = current_shape
#     if min_shape > current_shape: min_shape = current_shape
# print(min_shape, max_shape)

# import matplotlib.pyplot as plt
# plt.pie(stats.values(), labels=stats.keys())
# plt.show()



# import math

# df = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')
# df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)

# channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

# for i in range(2000):
#     mat = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2]
#     for j in range(12):
#         channels_of_12[j].append(mat[j])

# means = []
# stds = []
# for channel in channels_of_12:
#     counter = 0

#     regular_sum = 0
#     squared_sum = 0

#     for element in channel:
#         counter += len(element)
#         regular_sum += sum(element)

#     for element in channel:
#         squared_sum += sum(pow(element - regular_sum / counter, 2))

#     means.append(regular_sum / counter)
#     stds.append(math.sqrt(squared_sum / counter))

# print('means: ', means)
# print('stds: ', stds)