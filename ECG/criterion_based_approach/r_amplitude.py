import numpy as np
from criterion_based_approach.util import get_channel, get_values_ignoring_nan

def get_median_r_amplitude(ecg_signal, sampling_rate, ecg_parameters):
    ecg_signal_v4, _ = get_channel(ecg_signal, sampling_rate, 'V4')

    r_amplitudes = np.abs(ecg_signal_v4[ecg_parameters['R_peaks']] - get_values_ignoring_nan(ecg_signal_v4, ecg_parameters['P_offsets']))
    
    return np.nanmedian(r_amplitudes)