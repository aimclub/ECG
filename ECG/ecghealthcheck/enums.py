from enum import Enum


class ECGClass(Enum):
    NORM = 'norm'
    ALL = 'all'
    STTC = 'sttc'
    MI = 'mi'
    HYP = 'hyp'
    CD = 'cd'


class ECGStatus(Enum):
    NORM = 1
    ABNORM = 0
