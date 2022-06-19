import numpy as np
import neurokit2 as nk


def get_r_peaks(ecg_cleaned, sampling_rate):
    _, peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

    return np.array(peaks['ECG_R_Peaks'])


def get_pqst_peaks(ecg_cleaned, sampling_rate, r_peaks):
    _, peaks = nk.ecg_delineate(ecg_cleaned, sampling_rate=sampling_rate,
                                rpeaks=r_peaks, method='peaks', show=False)

    return np.array(peaks['ECG_P_Peaks']), np.array(peaks['ECG_Q_Peaks']),\
        np.array(peaks['ECG_S_Peaks']), np.array(peaks['ECG_T_Peaks'])


def get_p_offsets_s_offsets_q_onsets_t_offsets(ecg_cleaned, sampling_rate, r_peaks):
    _, waves_dwt = nk.ecg_delineate(ecg_cleaned, sampling_rate=sampling_rate,
                                    rpeaks=r_peaks, method='dwt', show=False)

    return np.array(waves_dwt['ECG_P_Offsets']), np.array(waves_dwt['ECG_R_Offsets']),\
        np.array(waves_dwt['ECG_R_Onsets']), np.array(waves_dwt['ECG_T_Offsets'])


def get_mean_rr(r_peaks):
    rr_lengths = []

    for i in range(len(r_peaks) - 1):
        rr_lengths.append(r_peaks[i + 1] - r_peaks[i])

    return round(np.nanmean(np.array(rr_lengths)))


def get_ecg_parameters(ecg_cleaned, sampling_rate):
    ecg_parameters = {}

    ecg_parameters['R_peaks'] = get_r_peaks(ecg_cleaned, sampling_rate)
    ecg_parameters['P_peaks'], ecg_parameters['Q_peaks'],\
        ecg_parameters['S_peaks'], ecg_parameters['T_peaks']\
        = get_pqst_peaks(ecg_cleaned, sampling_rate, ecg_parameters['R_peaks'])

    ecg_parameters['P_offsets'], ecg_parameters['S_offsets'],\
        ecg_parameters['Q_onsets'], ecg_parameters['T_offsets']\
        = get_p_offsets_s_offsets_q_onsets_t_offsets(ecg_cleaned, sampling_rate,
                                                     ecg_parameters['R_peaks'])

    ecg_parameters['RR'] = get_mean_rr(ecg_parameters['R_peaks'])

    return ecg_parameters
