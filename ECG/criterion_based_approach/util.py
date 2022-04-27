import numpy as np
import neurokit2 as nk
from math import isnan, nan

def samples_to_ms(value, sampling_rate):
    return value / sampling_rate * 1000


def mV_to_mm(value, mV_in_mm=10):
    return value * mV_in_mm


def get_values_ignoring_nan(array, indices):
    out = np.zeros(indices.shape[0])

    for i in range(indices.shape[0]):
        if isnan(indices[i]):
            out[i] = nan
        else:
            out[i] = array[int(indices[i])]

    return out


def get_channel(ecg_signal, sampling_rate, channel_name):
    if channel_name == '2':
        ecg_channel = ecg_signal[1, :]
    elif channel_name == 'V3':
        ecg_channel = ecg_signal[8, :]
    elif channel_name == 'V4':
        ecg_channel = ecg_signal[9, :]
    else:
        ecg_channel = np.zeros_like(ecg_signal[1, :])

    channel_cleaned = nk.ecg_clean(ecg_channel, sampling_rate=sampling_rate)

    return ecg_channel, channel_cleaned