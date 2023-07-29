import numpy as np
from ECG.ecghealthcheck.enums import ECGClass
from ECG.ecghealthcheck.signal_preprocessing import get_model
from ECG.ecghealthcheck.signal_preprocessing import filter_ecg
from ECG.ecghealthcheck.signal_preprocessing import ecg_to_tensor
from ECG.ecghealthcheck.signal_preprocessing import normalize_ecg


def ecg_is_normal(signal: np.ndarray, data_type: ECGClass) -> bool:
    model = get_model(data_type)

    signal = filter_ecg(signal)
    signal = normalize_ecg(signal)
    signal = ecg_to_tensor(signal)

    return model.predict(signal)
