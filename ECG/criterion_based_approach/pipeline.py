from criterion_based_approach.util import get_channel
from criterion_based_approach.ecg_features import get_ecg_parameters
from criterion_based_approach.qtc import get_median_qt, get_qtc
from criterion_based_approach.r_amplitude import get_median_r_amplitude
from criterion_based_approach.st_elevation import get_median_ste60
from criterion_based_approach.criterion import calculate_stemi_criterion, get_stemi_diagnosis

def process_recording(ecg_signal, sampling_rate):
    ## Peaks

    _, ecg_cleaned_2 = get_channel(ecg_signal, sampling_rate, '2')
    ecg_parameters = get_ecg_parameters(ecg_cleaned_2, sampling_rate)

    ## QTc

    median_qt = get_median_qt(ecg_cleaned_2, ecg_parameters)
    qtc_ms = get_qtc(median_qt, ecg_parameters['RR'], sampling_rate)

    ## RA

    median_r_amplitude = get_median_r_amplitude(ecg_signal, sampling_rate, ecg_parameters)

    ## STE60

    median_ste60 = get_median_ste60(ecg_signal, sampling_rate, ecg_parameters)

    ## STEMI criterion

    stemi_criterion = calculate_stemi_criterion(qtc_ms, median_r_amplitude, median_ste60)
    stemi_diagnosis = get_stemi_diagnosis(stemi_criterion)

    return stemi_diagnosis, stemi_criterion, qtc_ms, median_r_amplitude, median_ste60


def get_ste(ecg_signal, sampling_rate):
    _, ecg_cleaned_2 = get_channel(ecg_signal, sampling_rate, '2')
    ecg_parameters = get_ecg_parameters(ecg_cleaned_2, sampling_rate)

    return get_median_ste60(ecg_signal, sampling_rate, ecg_parameters)
