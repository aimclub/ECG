from criterion_based_approach.util import mV_to_mm

def calculate_stemi_criterion(qtc_ms, median_r_amplitude, median_ste60):
    ste60_mm = mV_to_mm(median_ste60)
    r_amplitude_mm = mV_to_mm(median_r_amplitude)

    return (1.196 * ste60_mm) + (0.059 * qtc_ms) - (0.326 * min(r_amplitude_mm, 15))


def get_stemi_diagnosis(stemi_criterion, threshold=28.13):
    return stemi_criterion > threshold