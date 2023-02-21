import pandas as pd
import os
import scipy 
import numpy as np
import torch
from Filtering.Neurokit2Filters import filter_ecg

class FewShotDataset:

    def __init__(self, shot=3):
        self.shot = shot

        self.df = pd.read_csv('Data\PTB-XL\Test\\test_labels.csv')

        if not os.path.exists(f'Data\PTB-XL\Test\TrainFewShot(Shot={self.shot})'):
            os.makedirs      (f'Data\PTB-XL\Test\TrainFewShot(Shot={self.shot})', exist_ok=True)
        if not os.path.exists(f'Data\PTB-XL\Test\TestFewShot(Shot={self.shot})'):
            os.makedirs      (f'Data\PTB-XL\Test\TestFewShot(Shot={self.shot})', exist_ok=True)

    
    def get_train_data(self):

        if len(os.listdir(f'Data\PTB-XL\Test\TrainFewShot(Shot={self.shot})')) == 0:

            ECGs = []
            for column_name in self.df.columns[1:]:
                df_with_column = self.df.loc[self.df[column_name] == 1]

                for i in range(self.shot):
                    ECGs.append(scipy.io.loadmat(
                        f'Data\PTB-XL\Test\Clean\{str(df_with_column.iloc[i]["ecg_id"]).zfill(5)}.mat')['ECG'][:3000, :]
                    )

            ECGs = self.prepare_ECG(ECGs)
            self.save_to_dir(f'Data\PTB-XL\Test\TrainFewShot(Shot={self.shot})', ECGs)
        else:
            ECGs = self.read_from_dir(f'Data\PTB-XL\Test\TrainFewShot(Shot={self.shot})')

        
        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i], dtype=torch.float32)


        diagnoses = []
        for diagnose_class in range(len(self.df.columns[1:])):
            for _ in range(self.shot):
                diagnoses.append(diagnose_class)


        return diagnoses, ECGs


    
    def get_test_data(self):

        if len(os.listdir(f'Data\PTB-XL\Test\TestFewShot(Shot={self.shot})')) == 0:

            ECGs = []
            for column_name in self.df.columns[1:]:
                df_with_column = self.df.loc[self.df[column_name] == 1]
                
                for i in range(self.shot, 260):
                    ECGs.append(scipy.io.loadmat(
                        f'Data\PTB-XL\Test\Clean\{str(df_with_column.iloc[i]["ecg_id"]).zfill(5)}.mat')['ECG'][:3000, :]
                    )

            ECGs = self.prepare_ECG(ECGs)
            self.save_to_dir(f'Data\PTB-XL\Test\TestFewShot(Shot={self.shot})', ECGs)

        else:
            ECGs = self.read_from_dir(f'Data\PTB-XL\Test\TestFewShot(Shot={self.shot})')

        
        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i], dtype=torch.float32)


        diagnoses = []
        for diagnose_class in range(len(self.df.columns[1:])):
            for _ in range(self.shot, 260):
                diagnoses.append(diagnose_class)


        return diagnoses, ECGs


    
    def prepare_ECG(self, ECGs):
        # Filtering
        for i in range(len(ECGs)):
            ECGs[i] = filter_ecg(ECGs[i])
            self.print_progressBar(i+1, len(ECGs), prefix='Filtering ECG: ', length=50)

        #Normalizing
        means, stds = self.get_channel_means_stds(ECGs)
        for i, ecg in enumerate(ECGs):
            for j in range(12):
                ecg[j] = (ecg[j] - means[j]) / stds[j]
            self.print_progressBar(i+1, len(ECGs), prefix='Normalizing ECG:', length=50)

        return ECGs
    
    def get_channel_means_stds(self, ECGs):

        ECGs = np.asarray(ECGs)
        means = ECGs.mean(axis=(0,2))
        stds = ECGs.std(axis=(0,2))

        return means, stds

    
    def save_to_dir(self, path, ECGs):
        for i, ECG in enumerate(ECGs):
            scipy.io.savemat(f'{path}\{i}.mat', {'ECG': ECG})

    def read_from_dir(self, path):
        ECGs = []
        for file in os.listdir(path):
            ECGs.append(scipy.io.loadmat(f'{path}\{file}')['ECG'])
        return ECGs


    def print_progressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
