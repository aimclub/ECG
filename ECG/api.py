from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from digitization.preprocessing import open_image, image_rotation, binarization
from digitization.digitization import grid_detection, signal_extraction


class Diagnosis(Enum):
    MI = 'Myocardial Infarction'
    BER = 'Benign Early Repolarization'


@dataclass
class RiskMarkers:
    Ste60_V3: float
    QTc: float
    RA_V4: float



def convert_image_to_signal(image: Image.Image) -> np.ndarray:
    image = open_image(image)
    rotated_image = image_rotation(image)
    scale = grid_detection(rotated_image)
    binary_image = binarization(rotated_image)
    ecg_signal = signal_extraction(binary_image, scale)

    return ecg_signal   

def check_ST_elevation(s: np.ndarray) -> bool:
    raise NotImplementedError()

def evaluate_risk_markers(s: np.ndarray) -> RiskMarkers:
    raise NotImplementedError()

def diagnose_with_STEMI(s: np.ndarray) -> Tuple[Diagnosis, str]:
    raise NotImplementedError()

def diagnose_with_NN(s: np.ndarray) -> Tuple[Diagnosis, Image.Image]:
    raise NotImplementedError()
