import numpy as np
from criterion_based_approach.util import get_channel, get_values_ignoring_nan

def get_j_points_neurokit(ecg_parameters):
    return ecg_parameters['S_offsets']


def get_j60_points(j_points, sampling_rate, ecg_length):
    return np.minimum(j_points + int(60 * sampling_rate / 1000), ecg_length - 1)


def get_median_ste60(ecg_signal, sampling_rate, ecg_parameters):
    ecg_signal_v3, ecg_cleaned_v3 = get_channel(ecg_signal, sampling_rate, 'V3')

    j_points = get_j_points_neurokit(ecg_parameters)
    j60_points = get_j60_points(j_points, sampling_rate, ecg_cleaned_v3.shape[0])

    ste60 = get_values_ignoring_nan(ecg_signal_v3, j60_points) - get_values_ignoring_nan(ecg_signal_v3, ecg_parameters['P_offsets'])
    median_ste60 = np.nanmedian(ste60)

    return median_ste60