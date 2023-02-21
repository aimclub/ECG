from torch.utils.data import Dataset
import random
import numpy as np
import os
import pandas as pd
import scipy.io
import torch
from  Datasets.Physionet.NoisyDatasetPreprocessing import prepare_dataset

class PairsDataset(Dataset):
    def __init__(self, WITH_ROLL=False):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.path_append = ''
        if WITH_ROLL: self.path_append = '_Rolled'

        if not os.path.exists(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}'):
            os.mkdir(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}')
        if len(os.listdir(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}')) == 0:
            prepare_dataset(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}\\')

        self.dataset_len = len(os.listdir(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}'))

    def __getitem__(self, index):
        if index % 2 == 0:
            index = int(index / 2)
            rand_index = int(np.random.randint(0, self.dataset_len) / 2)
            while rand_index == index: 
                rand_index = int(np.random.randint(0, self.dataset_len) / 2)

            ecg1 = scipy.io.loadmat(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}\{index}_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}\{rand_index}_clean.mat')['ECG']
            label = 0.
        else:
            index = int(index / 2)
            ecg1 = scipy.io.loadmat(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}\{index}_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\PTB-XL\Train\Prepared_Noisy{self.path_append}\{index}_noisy.mat')['ECG']
            label = 1.

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32),
            ), torch.as_tensor((label), dtype=torch.float32)

    def __len__(self):
        return int(self.dataset_len) #* 0.20)