from enum import Enum
from dataclasses import dataclass

class Diagnosis(Enum):
    MI = 'Myocardial Infarction'
    BER = 'Benign Early Repolarization'


@dataclass
class RiskMarkers:
    Ste60_V3: float
    QTc: float
    RA_V4: float
