import scipy
import torch
import numpy as np
import neurokit2 as nk
from typing import List, Tuple
from ECG.ecghealthcheck.utils import ECG_LENGTH
from ECG.ecghealthcheck.enums import ECGClass
from ECG.ecghealthcheck.utils import few_shot_files
from ECG.ecghealthcheck.utils import FILTER_METHOD
from ECG.ecghealthcheck.utils import normalization_params
from ECG.ecghealthcheck.models.classificator import Classificator


def get_few_shot_data(
        abnormal_type: ECGClass) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    normal_files = few_shot_files[ECGClass.NORM]
    abnormal_files = few_shot_files[abnormal_type]

    norm_ecgs = []
    abnorm_ecgs = []

    for norm, abnorm in zip(normal_files, abnormal_files):
        norm_ecgs.append(scipy.io.loadmat(
            f'ECG/ecghealthcheck/data/{norm}'
        )['ECG'][:, :ECG_LENGTH])
        abnorm_ecgs.append(scipy.io.loadmat(
            f'ECG/ecghealthcheck/data/{abnorm}'
        )['ECG'][:, :ECG_LENGTH])

    return norm_ecgs, abnorm_ecgs


def filter_ecg(ecg: np.ndarray):
    for lead in range(ecg.shape[0]):
        ecg[lead] = nk.ecg_clean(ecg[lead], sampling_rate=500, method=FILTER_METHOD)
    return ecg


def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    for lead in range(ecg.shape[0]):
        ecg[lead] = (ecg[lead] - normalization_params['mean'][lead]) / \
            normalization_params['std'][lead]
    return ecg


def ecg_to_tensor(ecg: np.ndarray):
    return torch.as_tensor(ecg, dtype=torch.float32)[None, :, :]


def get_model(data_type: ECGClass) -> Classificator:

    model = Classificator()
    norm_ecgs, abnorm_ecgs = get_few_shot_data(data_type)

    norm_ecgs = list(map(filter_ecg, norm_ecgs))
    abnorm_ecgs = list(map(filter_ecg, abnorm_ecgs))

    norm_ecgs = list(map(normalize_ecg, norm_ecgs))
    abnorm_ecgs = list(map(normalize_ecg, abnorm_ecgs))

    norm_ecgs = list(map(ecg_to_tensor, norm_ecgs))
    abnorm_ecgs = list(map(ecg_to_tensor, abnorm_ecgs))

    model.fit(norm_ecgs, abnorm_ecgs)

    return model
