from ECG.criterion_based_approach.util import get_channel
from ECG.criterion_based_approach.ecg_features import get_ecg_parameters
from ECG.criterion_based_approach.qtc import get_median_qt, get_qtc
from ECG.criterion_based_approach.r_amplitude import get_median_r_amplitude
from ECG.criterion_based_approach.st_elevation import get_median_ste60
from ECG.criterion_based_approach.criterion import calculate_stemi_criterion, \
    get_stemi_diagnosis
from ECG.data_classes import RiskMarkers


def detect_risk_markers(ecg_signal, sampling_rate) -> RiskMarkers:
    # Peaks

    _, ecg_cleaned_2 = get_channel(ecg_signal, sampling_rate, '2')
    ecg_parameters = get_ecg_parameters(ecg_cleaned_2, sampling_rate)

    # QTc

    median_qt = get_median_qt(ecg_cleaned_2, ecg_parameters)
    qtc_ms = get_qtc(median_qt, ecg_parameters['RR'], sampling_rate)

    # RA

    median_r_amplitude = get_median_r_amplitude(ecg_signal, sampling_rate, ecg_parameters)

    # STE60

    median_ste60 = get_median_ste60(ecg_signal, sampling_rate, ecg_parameters)

    return RiskMarkers(Ste60_V3=median_ste60, QTc=qtc_ms, RA_V4=median_r_amplitude)


def diagnose(risk_markers: RiskMarkers, tuned: bool):
    stemi_criterion = calculate_stemi_criterion(
        risk_markers.QTc, risk_markers.RA_V4, risk_markers.Ste60_V3, tuned)
    stemi_diagnosis = get_stemi_diagnosis(stemi_criterion, tuned)

    return stemi_diagnosis, stemi_criterion


def get_ste(ecg_signal, sampling_rate):
    _, ecg_cleaned_2 = get_channel(ecg_signal, sampling_rate, '2')
    ecg_parameters = get_ecg_parameters(ecg_cleaned_2, sampling_rate)

    return get_median_ste60(ecg_signal, sampling_rate, ecg_parameters)
