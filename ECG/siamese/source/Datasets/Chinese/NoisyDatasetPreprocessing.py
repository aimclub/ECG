import os
import pandas as pd
import scipy
from Filtering.Neurokit2Filters import filter_ecg
# from Filtering.PreprocessingFilters import filter_ecg
import numpy as np
import math

np.random.seed(42)

FRAGMENT_SIZE = 2900    # or KERNEL_SIZE
STEP_SIZE = 1500        # or STRIDE
df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)
dataset_size = len(df)
total_data = []

def prepare_dataset(path='Data\ChineseDataset\PreparedDataset_Noisy\\'):


    if not os.path.exists('Data\ChineseDataset\FilteredECG') or len(os.listdir('Data\ChineseDataset\FilteredECG')) == 0:
        os.mkdir('Data\ChineseDataset\FilteredECG')
        for i in range(dataset_size):
            ecg = scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2]
        
            ### Filtering EKG
            ecg = filter_ecg(ecg)

            scipy.io.savemat(f'Data\ChineseDataset\FilteredECG\{df["Recording"][i]}.mat', {'ECG': ecg})

            recording = [ecg, df['Recording'][i]]        
            total_data.append(recording)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)


    else:
        for i in range(dataset_size):
            ecg = scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{df["Recording"][i]}.mat')['ECG']

            recording = [ecg, df['Recording'][i]]
            total_data.append(recording)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)

    print("Filtering done! Starting channel-wise ECG normalization...")


    ## Channel-wise normalization
    channel_means, channel_stds = get_channel_means_stds(total_data)
    for i, recording in enumerate(total_data):
        for j in range(12):
            recording[0][j] = (recording[0][j] - channel_means[j]) / channel_stds[j]
        print_progressBar(i+1, dataset_size, prefix='Normalizing ECG:', length=50)

    print(f"Normaization done! Saving data to {path}")


    bias = 0
    for i, recording in enumerate(total_data):
        total_fragments_in_ecg = math.floor((len(recording[0][0]) - FRAGMENT_SIZE) / STEP_SIZE) + 1 # Convolution formula
        
        for fragment_index in range(total_fragments_in_ecg):
            scipy.io.savemat(f'{path}{bias + fragment_index}_clean.mat', {'ECG': recording[0][:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE]})
            
            #Gaussian noise
            noise = np.random.normal(0, channel_stds[0]*0.25, [1, FRAGMENT_SIZE])
            for j in range(1, 12):
                noise = np.concatenate((noise, np.random.normal(0, channel_stds[j]*0.25, [1, FRAGMENT_SIZE])), axis=0)

            #Baseline wander
            L = FRAGMENT_SIZE
            x = np.linspace(0, L, L)
            A = np.random.uniform(0.1, 1.)
            T = L # 2 * L
            PHI = np.random.uniform(0, 2 * math.pi)
            wander = []
            for j in x:
                wander.append(A * np.cos(2 * math.pi * (j/T) + PHI))
            noise = np.sum([noise, np.array(wander)], axis=0)

            if 'Rolled' in path:
                roll = np.random.randint(-1500, 1500)
                scipy.io.savemat(f'{path}{bias + fragment_index}_noisy.mat', {'ECG': np.roll(recording[0][:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE] + noise, roll, axis=1)})
            else:
                scipy.io.savemat(f'{path}{bias + fragment_index}_noisy.mat', {'ECG': recording[0][:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE] + noise})
        
        bias += total_fragments_in_ecg

        print_progressBar(i+1, dataset_size, prefix='Saving:', length=50)

    print("Dataset preparation complete!")


def get_channel_means_stds(total_data):

    channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for recording in total_data:
        for j in range(12):
            channels_of_12[j].append(recording[0][j])

    means = []
    stds = []
    for channel in channels_of_12:
        
        counter = 0
        regular_sum = 0
        squared_sum = 0

        for ecg_lead in channel:
            counter += len(ecg_lead)
            regular_sum += sum(ecg_lead)
        for ecg_lead in channel:
            squared_sum += sum(pow(ecg_lead - regular_sum / counter, 2))

        means.append(regular_sum / counter)
        stds.append(math.sqrt(squared_sum / counter))

    return means, stds

def get_channel_mins_maxs(total_data):
    channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for recording in total_data:
        for j in range(12):
            channels_of_12[j].append(recording[0][j])

    mins = []
    maxs = []

    for channel in channels_of_12:
        loc_mins = []
        loc_maxs = []
        for ecg in channel:
            loc_mins.append(np.min(ecg))
            loc_maxs.append(np.max(ecg))
        mins.append(np.min(loc_mins))
        maxs.append(np.min(loc_maxs))
    
    return mins, maxs

def print_progressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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