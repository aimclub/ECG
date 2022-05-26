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
    ste_bool, ste_mV, threshold, explanation = api.check_ST_elevation(signal, sampling_rate)
    expected_ste_mV = 0.225
    expected_ste_bool = True

    assert ste_mV == expected_ste_mV, f"Failed to predict ST elevation value in mV: expected {expected_ste_mV}, got {ste_mV}"
    assert ste_bool == expected_ste_bool, "Failed to recognize significant ST elevation"

    expected_explanation = "ST elevation value in lead V3 (0.225 mV) exceeded the threshold 0.2, therefore ST elevation was detected."
    assert explanation == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {explanation}"


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

    stemi_positive_tuned = api.diagnose_with_STEMI(signal_stemi, sampling_rate, True)
    stemi_negative_tuned = api.diagnose_with_STEMI(signal_er, sampling_rate, True)

    # positive tuned
    assert stemi_positive_tuned[0] == Diagnosis.MI, "Failed to recognize MI"
    expected_explanation = "Criterion value calculated as follows: (2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms]) + (-1.7 * np.minimum([RA V4 in mm], 19)) = 151.47 exceeded the threshold 126.9, therefore the diagnosis is Myocardial Infarction"
    assert stemi_positive_tuned[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {stemi_positive_tuned[1]}"

    # negative tuned
    assert stemi_negative_tuned[0] == Diagnosis.BER, "Failed to recognize BER"
    expected_explanation = "Criterion value calculated as follows: (2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms]) + (-1.7 * np.minimum([RA V4 in mm], 19)) = 118.4062869471591 did not exceed the threshold 126.9, therefore the diagnosis is Benign Early Repolarization"
    assert stemi_negative_tuned[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {stemi_negative_tuned[1]}"

    stemi_positive_original = api.diagnose_with_STEMI(signal_stemi, sampling_rate, False)

    # positive original explanation
    assert stemi_positive_original[0] == Diagnosis.MI, "Failed to recognize MI"
    expected_explanation = "Criterion value calculated as follows: (1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms]) – (0.326 * [RA V4 in mm])) = 31.2231 exceeded the threshold 23.4, therefore the diagnosis is Myocardial Infarction"
    assert stemi_positive_original[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {stemi_positive_original[1]}"


def test_diagnose_with_NN_test():
    filename_not_ber = './ECG/tests/test_data/NotBER.mat'
    filename_er = './ECG/tests/test_data/BER.mat'
    signal_not_ber = get_ecg_signal(filename_not_ber)
    signal_er = get_ecg_signal(filename_er)

    ber_positive = api.diagnose_with_NN(signal_er)
    ber_negative = api.diagnose_with_NN(signal_not_ber)

    assert ber_positive[0] == Diagnosis.BER, "Failed to recognize BER"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.8727"
    assert ber_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_positive[1]}"

    assert ber_negative[0] == Diagnosis.Unknown, f"Wrong explanation\n\tGot {ber_negative[0]}"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.5973"
    assert ber_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_negative[1]}"
