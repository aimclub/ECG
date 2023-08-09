import numpy as np
from ECG.condition.enums import Condition
from ECG.condition.signal_preprocessing import get_model
from ECG.condition.signal_preprocessing import filter_ecg
from ECG.condition.signal_preprocessing import ecg_to_tensor
from ECG.condition.signal_preprocessing import normalize_ecg


def is_condition_present(signal: np.ndarray, condition: Condition) -> bool:
    model = get_model(condition)

    signal = filter_ecg(signal)
    signal = normalize_ecg(signal)
    signal = ecg_to_tensor(signal)

    return ~model.predict(signal)
