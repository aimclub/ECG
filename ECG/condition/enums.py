from enum import Enum


class Condition(Enum):
    ANY = 'all'
    STTC = 'sttc'
    MI = 'mi'


class ECGStatus(Enum):
    NORM = 1
    ABNORM = 0
