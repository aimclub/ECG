import os
from torch.utils.data import Dataset
from Datasets.Chinese.NoisyDatasetPreprocessing import prepare_dataset
import scipy
import math
import torch
import random
import numpy as np

class NoisyPairsDataset(Dataset):
    
    def __init__(self, WITH_ROLL=False):

        random.seed(42)
        np.random.seed(42)

        self.path_append = ''
        if WITH_ROLL: self.path_append = '_Rolled'

        if (not os.path.exists(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')):
            os.mkdir(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')
        if (len(os.listdir(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')) == 0):
            prepare_dataset(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\\')

        self.dataset_len = len(os.listdir(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}'))
        
    def __getitem__(self, index):
        if index % 2 == 0:
            index = int(index / 2)
            rand_index = int(np.random.randint(0, self.dataset_len) / 2)
            while rand_index == index: 
                rand_index = int(np.random.randint(0, self.dataset_len) / 2)

            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{index}_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{rand_index}_clean.mat')['ECG']
            label = 0.
        else:
            index = int(index / 2)
            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{index}_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{index}_noisy.mat')['ECG']
            label = 1.

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32),
            ), torch.as_tensor((label), dtype=torch.float32)

    def __len__(self):
        return self.dataset_len