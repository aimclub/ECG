from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from criterion_based_approach.pipeline import process_recording, get_ste


class Diagnosis(Enum):
    MI = 'Myocardial Infarction'
    BER = 'Benign Early Repolarization'


@dataclass
class RiskMarkers:
    Ste60_V3: float
    QTc: float
    RA_V4: float



def convert_image_to_signal(image: Image.Image) -> np.ndarray:
    raise NotImplementedError()


def check_ST_elevation(signal: np.ndarray, sampling_rate: int) -> bool:
    return get_ste(signal, sampling_rate)


def evaluate_risk_markers(signal: np.ndarray, sampling_rate: int) -> RiskMarkers:
    _, _, qtc, r_amplitude, ste60 = process_recording(signal, sampling_rate)

    return RiskMarkers(Ste60_V3=ste60, QTc=qtc, RA_V4=r_amplitude)


def diagnose_with_STEMI(signal: np.ndarray, sampling_rate: int) -> Tuple[Diagnosis, str]:
    stemi_diagnosis, stemi_criterion, _, _, _ = process_recording(signal, sampling_rate)
    diagnosis_enum = Diagnosis.MI if stemi_diagnosis else Diagnosis.BER
    explanation = 'Criterion value calculated as follows: ' + \
        '(1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms]) - (0.326 * min([RA V4 in mm], 15)) = ' + str(stemi_criterion) + \
        (' exceeded ', ' did not exceed ')[stemi_diagnosis] + \
        'the threshold 28.13, therefore the diagnosis is ' + diagnosis_enum.value

    return (diagnosis_enum, explanation)


def diagnose_with_NN(s: np.ndarray) -> Tuple[Diagnosis, Image.Image]:
    raise NotImplementedError()
