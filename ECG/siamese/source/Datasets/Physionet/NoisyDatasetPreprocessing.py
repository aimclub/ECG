import os
import pandas as pd
import scipy.io
from Filtering.Neurokit2Filters import filter_ecg
# from Filtering.PreprocessingFilters import filter_ecg
import numpy as np
import math

np.random.seed(42)

FRAGMENT_SIZE = 3000   # or KERNEL_SIZE
STEP_SIZE = 2000       # or STRIDE
df = pd.read_csv('Data\PTB-XL\Train\\train_labels.csv', delimiter=',')
dataset_size = len(df)
total_data = []

def prepare_dataset(path='Data\ChineseDataset\PreparedDataset_Noisy\\'):
    
    if not os.path.exists('Data\PTB-XL\Train\Filtered') or len(os.listdir('Data\PTB-XL\Train\Filtered')) == 0:
        os.mkdir('Data\PTB-XL\Train\Filtered')

        for i in range(dataset_size):
            ecg = scipy.io.loadmat(f'Data\PTB-XL\Train\Clean\{str(df["ecg_id"][i]).zfill(5)}.mat')['ECG']
            ecg = np.transpose(ecg)

            ### Filtering EKG
            ecg = filter_ecg(ecg)

            scipy.io.savemat(f'Data\PTB-XL\Train\Filtered\{str(df["ecg_id"][i]).zfill(5)}.mat', {'ECG': ecg})

            # recording = [ecg, df['Recording'][i]]        
            total_data.append(ecg)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)


    else:
        for i in range(dataset_size):
            ecg = scipy.io.loadmat(f'Data\PTB-XL\Train\Filtered\{str(df["ecg_id"][i]).zfill(5)}.mat')['ECG']

            # recording = [ecg, df['Recording'][i]]
            total_data.append(ecg)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)

    print("Filtering done! Starting channel-wise ECG normalization...")


    ## Channel-wise normalization
    channel_means, channel_stds = get_channel_means_stds(total_data)
    # channel_means = [-3.718357624889972e-06, -5.319391209417234e-05, -4.945751864238487e-05, 2.7634547420381818e-05, 2.2198677172505495e-05, -4.8649972926477536e-05, -5.057949826273172e-05, -0.00011746589092615135, -3.591956367084942e-05, -2.8080299717502795e-05, -7.547788461059586e-06, 3.886816171659815e-05]
    # channel_stds = [0.098066450039095, 0.15970448599478773, 0.1160161432676049, 0.11918654198060893, 0.07185394529804712, 0.1305917404821007, 0.12825125933221615, 0.28536948725346756, 0.27532385031389156, 0.24518165680480666, 0.187966083727839, 0.14920102104532423]
    for i, ecg in enumerate(total_data):
        for j in range(12):
            ecg[j] = (ecg[j] - channel_means[j]) / channel_stds[j]
        print_progressBar(i+1, dataset_size, prefix='Normalizing ECG:', length=50)

    print(f"Normaization done! Saving data to {path}")


    bias = 0
    for i, ecg in enumerate(total_data):
        # total_fragments_in_ecg = math.floor((len(recording[0][0]) - FRAGMENT_SIZE) / STEP_SIZE) + 1 # Convolution formula
        total_fragments_in_ecg = 2
        
        for fragment_index in range(total_fragments_in_ecg):
            scipy.io.savemat(f'{path}{bias + fragment_index}_clean.mat', {'ECG': ecg[:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE]})
            
            #Gaussian noise
            noise = np.random.normal(0, channel_stds[0]*0.1, [1, FRAGMENT_SIZE])
            for j in range(1, 12):
                noise = np.concatenate((noise, np.random.normal(0, channel_stds[j]*0.1, [1, FRAGMENT_SIZE])), axis=0)

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
                scipy.io.savemat(f'{path}{bias + fragment_index}_noisy.mat', {'ECG': np.roll(ecg[:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE] + noise, roll, axis=1)})
            else:
                scipy.io.savemat(f'{path}{bias + fragment_index}_noisy.mat', {'ECG': ecg[:, STEP_SIZE * fragment_index:STEP_SIZE * fragment_index + FRAGMENT_SIZE] + noise})
        
        bias += total_fragments_in_ecg

        print_progressBar(i+1, dataset_size, prefix='Saving:', length=50)

    print("Dataset preparation complete!")


def get_channel_means_stds(total_data):

    total_data = np.asarray(total_data)
    means = total_data.mean(axis=(0,2))
    stds = total_data.std(axis=(0,2))

    return means, stds



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