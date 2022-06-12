import torch.nn as nn
import numpy as np
import torch
from typing import Tuple
from ECG.NN_based_approach.utils import signal_rescale
from ECG.data_classes import Diagnosis
from ECG.NN_based_approach.model_factory import create_model
from ECG.NN_based_approach.NN_Enums import NetworkType, ModelType


def process_recording(signal: np.ndarray, net: nn.Module) -> np.double:
    x_val = torch.from_numpy(signal).float().reshape((1, 1, signal.shape[0], signal.shape[1]))
    with torch.no_grad():
        pred = net.forward(x_val)
    return round(pred.cpu().detach().item(), 4)


def diagnose_BER(signal: np.ndarray) -> Tuple[Diagnosis, str]:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.BER)
    result = process_recording(signal, net=net)
    diagnosis = Diagnosis.BER if result > 0.6 else Diagnosis.Unknown
    explanation = 'Neutal Network calculated: the probability of BER is ' + str(result)
    return (diagnosis, explanation)


def diagnose_MI(signal: np.ndarray) -> Tuple[Diagnosis, str]:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.MI)
    result = process_recording(signal, net=net)
    diagnosis = Diagnosis.MI if result > 0.6 else Diagnosis.Unknown
    explanation = 'Neutal Network calculated: the probability of MI is ' + str(result)
    return (diagnosis, explanation)


def diagnose_STE(signal: np.ndarray) -> Tuple[Diagnosis, str]:
    signal = signal_rescale(signal, up_slice=4000)
    net = create_model(net_type=NetworkType.Conv1, model_type=ModelType.STE, input_shape=(12, 4000))
    result = process_recording(signal, net=net)
    diagnosis = Diagnosis.STE if result > 0.6 else Diagnosis.Unknown
    explanation = 'Neutal Network calculated: the probability of STE is ' + str(result)
    return (diagnosis, explanation)
