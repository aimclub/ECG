import numpy as np
from ECG.criterion_based_approach.util import get_channel, get_values_ignoring_nan


def get_median_r_amplitude(ecg_signal, sampling_rate, ecg_parameters):
    ecg_signal_v4, _ = get_channel(ecg_signal, sampling_rate, 'V4')
    peaks = ecg_signal_v4[ecg_parameters['R_peaks']]
    offsets = get_values_ignoring_nan(ecg_signal_v4, ecg_parameters['P_offsets'])
    r_amplitudes = np.abs(peaks - offsets)
    return np.nanmedian(r_amplitudes)
