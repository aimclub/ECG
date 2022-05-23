import numpy as np
import ECG.api as api
from ECG.data_classes import Diagnosis
from ECG.tests.test_util import get_ecg_signal, get_ecg_array, open_image

def test_convert_image_to_signal():
    image_filename = './ECG/tests/test_data/ecg_image.jpg'
    array_filename = './ECG/tests/test_data/ecg_array.npy'

    signal = get_ecg_array(array_filename)
    image = open_image(image_filename)
    result = api.convert_image_to_signal(image)
    max_diff = np.abs(signal - result).max()

    assert max_diff < 1e-14, f'Recognized signal does not match the original. Max difference is {max_diff}'


def test_check_ST():
    filename = './ECG/tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename)
    ste = api.check_ST_elevation(signal, sampling_rate)
    ste_expected = 0.225
    assert ste == ste_expected, f"Failed to predict ST probability: expected {ste_expected}, got {ste}"


def test_evaluate_risk_markers():
    filename = './ECG/tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename)

    risk_markers = api.evaluate_risk_markers(signal, sampling_rate)
    expected = 0.225
    assert risk_markers.Ste60_V3 == expected, f"Failed to predict STE60 V3: expected {expected}, got {risk_markers.Ste60_V3}"
    expected = 501
    assert risk_markers.QTc == expected, f"Failed to predict QTc: expected {expected}, got {risk_markers.QTc}"
    expected = 0.315
    assert risk_markers.RA_V4 == expected, f"Failed to predict RA V4: expected {expected}, got {risk_markers.RA_V4}"


def test_diagnose_with_STEMI():
    filename_stemi = './ECG/tests/test_data/MI.mat'
    filename_er = './ECG/tests/test_data/BER.mat'
    sampling_rate = 500
    signal_stemi = get_ecg_signal(filename_stemi)
    signal_er = get_ecg_signal(filename_er)

    stemi_positive = api.diagnose_with_STEMI(signal_stemi, sampling_rate)
    stemi_negative = api.diagnose_with_STEMI(signal_er, sampling_rate)

    # positive
    assert stemi_positive[0] == Diagnosis.MI, "Failed to recognize MI"
    expected_explanation = "Criterion value calculated as follows: (1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms]) - (0.326 * min([RA V4 in mm], 15)) = 31.2231 did not exceed the threshold 28.13, therefore the diagnosis is Myocardial Infarction"
    assert stemi_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {stemi_positive[1]}"

    # negative
    assert stemi_negative[0] == Diagnosis.BER, "Failed to recognize BER"
    expected_explanation = "Criterion value calculated as follows: (1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms]) - (0.326 * min([RA V4 in mm], 15)) = 26.3252135133801 exceeded the threshold 28.13, therefore the diagnosis is Benign Early Repolarization"
    assert stemi_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {stemi_negative[1]}"
    print('***\n')

def diagnose_with_NN_test():
    filename_not_ber = './ECG/tests/test_data/NotBER.mat'
    filename_er = './ECG/tests/test_data/BER.mat'
    signal_not_ber = get_ecg_signal(filename_not_ber)
    signal_er = get_ecg_signal(filename_er)

    ber_positive = api.diagnose_with_NN(signal_er)
    ber_negative = api.diagnose_with_NN(signal_not_ber)

    assert ber_positive[0] == Diagnosis.BER, "Failed to recognize BER"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.8726784586906433"
    assert ber_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_positive[1]}"

    assert ber_negative[0] != Diagnosis.BER, f"Wrong explanation\n\tGot {ber_negative[0]}"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.5972990989685059"
    assert ber_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_negative[1]}"
    print('***\n')
