from ECG.criterion_based_approach.pipeline import get_ste
from tests.test_util import get_ecg_signal, compare_values


def test_check_ST_elevation():
    filename = './tests/test_data/MI.mat'
    signal = get_ecg_signal(filename)
    result = get_ste(signal, sampling_rate=500)
    compare_values(result, 0.225, "Failed to predict ST elevation value iv mV")
