from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


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

def evaluate_risk_markers(s: np.ndarray) -> RiskMarkers:
    raise NotImplementedError()

def diagnose_with_STEMI(s: np.ndarray) -> Tuple[Diagnosis, str]:
    raise NotImplementedError()

def diagnose_with_NN(s: np.ndarray) -> Tuple[Diagnosis, Image.Image]:
    raise NotImplementedError()
