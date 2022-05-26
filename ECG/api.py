from PIL import Image
import numpy as np
from typing import Tuple
from ECG.criterion_based_approach.pipeline import detect_risk_markers, diagnose, get_ste
from ECG.data_classes import Diagnosis, RiskMarkers
from ECG.digitization.preprocessing import image_rotation, binarization
from ECG.digitization.digitization import grid_detection, signal_extraction
from ECG.NN_based_approach.pipeline import process_recording, create_model
from ECG.NN_based_approach.NNType import NNType

def convert_image_to_signal(image: Image.Image) -> np.ndarray:
    image = np.asarray(image)
    rotated_image = image_rotation(image)
    scale = grid_detection(rotated_image)
    binary_image = binarization(rotated_image)
    ecg_signal = signal_extraction(binary_image, scale)

    return ecg_signal 

def check_ST_elevation(signal: np.ndarray, sampling_rate: int) -> Tuple[bool, float, str]:
    elevation_threshold = 0.2
    ste_mV = get_ste(signal, sampling_rate)
    ste_bool = ste_mV > elevation_threshold

    explanation = 'ST elevation value in lead V3 (' + str(ste_mV) + ' mV)' + (' did not exceed ', ' exceeded ')[ste_bool] + \
        'the threshold ' + str(elevation_threshold) + ', therefore ST elevation was' + (' not detected.', ' detected.')[ste_bool]

    return (ste_bool, ste_mV, elevation_threshold, explanation)


def evaluate_risk_markers(signal: np.ndarray, sampling_rate: int) -> RiskMarkers:
    return detect_risk_markers(signal, sampling_rate)


def diagnose_with_STEMI(signal: np.ndarray, sampling_rate: int, tuned: bool = False) -> Tuple[Diagnosis, str]:
    risk_markers = evaluate_risk_markers(signal, sampling_rate)
    stemi_diagnosis, stemi_criterion = diagnose(risk_markers, tuned)

    diagnosis_enum = Diagnosis.MI if stemi_diagnosis else Diagnosis.BER

    if tuned:
        formula = '(2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms]) + (-1.7 * np.minimum([RA V4 in mm], 19)) = '
        threshold = '126.9'
    else:
        formula = '(1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms]) – (0.326 * [RA V4 in mm])) = '
        threshold = '23.4'

    explanation = 'Criterion value calculated as follows: ' + \
        formula + str(stemi_criterion) + \
        (' did not exceed ', ' exceeded ')[stemi_diagnosis] + \
        'the threshold ' + threshold + ', therefore the diagnosis is ' + diagnosis_enum.value

    return (diagnosis_enum, explanation)


def diagnose_with_NN(signal: np.ndarray) -> Tuple[Diagnosis, str]:
    net = create_model(net_type=NNType.Conv)
    result = process_recording(signal, net=net)
    diagnosis = Diagnosis.BER if result > 0.7 else Diagnosis.Unknown
    explanation = 'Neutal Network calculated: the probability of BER is ' + str(result)
    return (diagnosis, explanation)
