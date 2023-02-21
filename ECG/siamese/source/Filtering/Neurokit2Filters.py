import neurokit2 as nk
import numpy as np

METHOD = "neurokit"

def filter_ecg(ecg):
    filtered_ecg = []
    for i in range(12):
        filtered_ecg.append(nk.ecg_clean(ecg[i], sampling_rate=500, method=METHOD))
    return np.asarray(filtered_ecg)