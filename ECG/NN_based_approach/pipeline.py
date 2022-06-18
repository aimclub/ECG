import torch.nn as nn
import numpy as np
import torch
from typing import Tuple
from ECG.NN_based_approach.utils import signal_rescale
from ECG.data_classes import ElevatedST
from ECG.NN_based_approach.model_factory import create_model
from ECG.NN_based_approach.NN_Enums import NetworkType, ModelType


def process_recording(signal: np.ndarray, net: nn.Module) -> np.double:
    x_val = torch.from_numpy(signal).float().reshape((1, 1, signal.shape[0], signal.shape[1]))
    with torch.no_grad():
        pred = net.forward(x_val)
    return round(pred.cpu().detach().item(), 4)


def is_BER(signal: np.ndarray) -> Tuple[bool, str]:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.BER)
    result = process_recording(signal, net=net)
    explanation = 'Neutal Network calculated: the probability of BER is ' + str(result)
    return (result > 0.6, explanation)


def is_MI(signal: np.ndarray) -> Tuple[bool, str]:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.MI)
    result = process_recording(signal, net=net)
    explanation = 'Neutal Network calculated: the probability of MI is ' + str(result)
    return (result > 0.6, explanation)


def check_STE(signal: np.ndarray) -> Tuple[ElevatedST, str]:
    signal = signal_rescale(signal, up_slice=4000)
    net = create_model(net_type=NetworkType.Conv1, model_type=ModelType.STE, input_shape=(12, 4000))
    result = process_recording(signal, net=net)
    explanation = 'Neutal Network calculated: the probability of significant ST elevation is ' + str(result)
    ste = ElevatedST.Present if result > 0.6 else ElevatedST.Abscent
    return (ste, explanation)
