import numpy as np
from ECG.ECG_embedding_classification.Enums import ECGStatus
from ECG.ECG_embedding_classification.Enums import ECGClass
from ECG.ECG_embedding_classification.preprocessing import get_model
from ECG.ECG_embedding_classification.preprocessing import filter_ecg
from ECG.ECG_embedding_classification.preprocessing import ecg_to_tensor
from ECG.ECG_embedding_classification.preprocessing import normalize_ecg


def classify_signal(signal: np.ndarray, data_type: ECGClass) -> ECGStatus:
    model = get_model(data_type)

    signal = filter_ecg(signal)
    signal = normalize_ecg(signal)
    signal = ecg_to_tensor(signal)

    return model.predict(signal)
