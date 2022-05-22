from ECG.criterion_based_approach.util import mV_to_mm
import numpy as np

def calculate_stemi_criterion(qtc_ms, median_r_amplitude, median_ste60):
    ste60_mm = mV_to_mm(median_ste60)
    r_amplitude_mm = mV_to_mm(median_r_amplitude)

    return (2.9 * ste60_mm) + (0.3 * qtc_ms) + (-1.7 * np.minimum(r_amplitude_mm, 19))


def get_stemi_diagnosis(stemi_criterion, threshold=126.9):
    return stemi_criterion >= threshold