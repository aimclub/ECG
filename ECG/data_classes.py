from enum import Enum
from dataclasses import dataclass
from PIL import Image


@dataclass
class Failed:
    reason: str


@dataclass
class TextExplanation:
    content: str


@dataclass
class NNExplanation:
    prob: float
    text: str
    images: [Image.Image]


@dataclass
class NNResult:
    prob: float
    images: [Image.Image]


class Diagnosis(Enum):
    STE = 'ST Elevation'
    MI = 'Myocardial Infarction'
    BER = 'Benign Early Repolarization'


class ElevatedST(Enum):
    Abscent = 'No significant ST elevation'
    Present = 'Significant ST elevation'


@dataclass
class RiskMarkers:
    """
    Store evaluated ECG risk markers

    Attributes:
        Ste60_V3 (float): ST segment elevation in lead V3, mm
        QTc (float): The corrected QT interval, ms
        RA_V4 (float): R-peak amplitude in lead V4, mm
    """
    Ste60_V3: float
    QTc: float
    RA_V4: float
