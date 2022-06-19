from ECG.criterion_based_approach.util import mV_to_mm
import numpy as np


def calculate_stemi_criterion(qtc_ms, median_r_amplitude, median_ste60, tuned: bool):
    ste60_mm = mV_to_mm(median_ste60)
    r_amplitude_mm = mV_to_mm(median_r_amplitude)

    if tuned:
        return (2.9 * ste60_mm) + (0.3 * qtc_ms) + (-1.7 * np.minimum(r_amplitude_mm, 19))
    else:
        return (1.196 * ste60_mm) + (0.059 * qtc_ms) - (0.326 * r_amplitude_mm)


def get_stemi_diagnosis(stemi_criterion, tuned: bool):
    if tuned:
        return stemi_criterion >= 126.9
    else:
        return stemi_criterion > 23.4
