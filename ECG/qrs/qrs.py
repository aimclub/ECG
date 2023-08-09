import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt


def _get_peaks(channel, sampling_rate):
    _, rpeaks = nk.ecg_peaks(channel, sampling_rate=sampling_rate)
    _, waves_peak = nk.ecg_delineate(
        channel, rpeaks, sampling_rate=sampling_rate, method="peak")
    peaks = {wave: waves_peak[f'ECG_{wave}_Peaks'] for wave in 'PQST'}
    peaks['R'] = rpeaks['ECG_R_Peaks'].tolist()
    return peaks


def _clean_signal(signal, sampling_rate):
    return np.stack([nk.ecg_clean(channel, sampling_rate=sampling_rate)
                     for channel in signal], axis=0)


def get_qrs(signal, sampling_rate):
    cleaned_signal = _clean_signal(signal, sampling_rate)
    peaks = [_get_peaks(channel, sampling_rate) for channel in cleaned_signal]
    return cleaned_signal, peaks


def _show_qrs(ax, channel, peaks):
    ax.plot(channel)

    waves = 'PQRST'
    colors = 'ygrbc'

    for (w, c) in zip(waves, colors):
        xs = peaks[w]
        params = dict(color=c, linestyle='--', linewidth=1)
        for i, x in enumerate(xs):
            if i == 0:
                ax.axvline(x=x, label=w, **params)
            else:
                ax.axvline(x=x, **params)

    ax.legend()


def show_channel_qrs(channel, channel_peaks):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    _show_qrs(ax, channel, channel_peaks)
    plt.close()
    return fig


def show_full_qrs(signal, peaks):
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(15, 18))
    for i in range(12):
        _show_qrs(axs[i // 2, i % 2], signal[i], peaks[i])
    plt.close()
    return fig
