from enum import Enum


class NetworkType(Enum):
    Conv = 'Conv'
    Conv1 = 'Conv1'


class ModelType(Enum):
    BER = 'ber'
    STE = 'ste'
    MI = 'mi'
