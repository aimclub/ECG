import numpy as np
import ECG.api as api
from ECG.criterion_based_approach.pipeline import get_ste
from ECG.data_classes import Diagnosis, ElevatedST
from tests.test_util import get_ecg_signal, get_ecg_array, open_image

def test_convert_image_to_signal():
    image_filename = './tests/test_data/ecg_image.jpg'
    array_filename = './tests/test_data/ecg_array.npy'

    signal = get_ecg_array(array_filename)
    image = open_image(image_filename)
    result = api.convert_image_to_signal(image)
    max_diff = np.abs(signal - result).max()

    assert max_diff < 1e-14, f'Recognized signal does not match the original. Max difference is {max_diff}'


def test_check_ST():
    filename_ok = './tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename_ok)
    ste_assessment, explanation = api.check_ST_elevation(signal, sampling_rate)
    ste_mV = get_ste(signal, sampling_rate)

    expected_ste_mV = 0.225
    expected_ste_assessment = ElevatedST.Present

    # OK
    assert ste_mV == expected_ste_mV, f"Failed to predict ST elevation value in mV: expected {expected_ste_mV}, got {ste_mV}"
    assert ste_assessment == expected_ste_assessment, "Failed to recognize significant ST elevation"

    expected_explanation = "ST elevation value in lead V3 (0.225 mV) exceeded the threshold 0.2, therefore ST elevation was detected."
    assert explanation == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {explanation}"

    # Fails
    filename_fail = './tests/test_data/NeurokitFails.mat'
    ste_assessment_fail, explanation_fail = api.check_ST_elevation(get_ecg_signal(filename_fail), sampling_rate)
    
    expected_ste_assessment_fail = ElevatedST.Failed
    assert ste_assessment_fail == expected_ste_assessment_fail, "Failed to handle an error while assessing ST elevation"

    expected_explanation_fail = "Failed to assess ST elevation due to an internal error"
    assert explanation_fail == expected_explanation_fail, f"Wrong explanation: \n\tExpected {expected_explanation_fail} \n\tGot {explanation_fail}"


def test_evaluate_risk_markers():
    filename_ok = './tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename_ok)

    # OK
    risk_markers = api.evaluate_risk_markers(signal, sampling_rate)
    expected = 0.225
    assert risk_markers.Ste60_V3 == expected, f"Failed to predict STE60 V3: expected {expected}, got {risk_markers.Ste60_V3}"
    expected = 501
    assert risk_markers.QTc == expected, f"Failed to predict QTc: expected {expected}, got {risk_markers.QTc}"
    expected = 0.315
    assert risk_markers.RA_V4 == expected, f"Failed to predict RA V4: expected {expected}, got {risk_markers.RA_V4}"

    # Fails
    filename_fail = './tests/test_data/NeurokitFails.mat'
    risk_markers_fail = api.evaluate_risk_markers(get_ecg_signal(filename_fail), sampling_rate)
    assert risk_markers_fail is None, f"Failed to handle an error while evaluating risk markers"


def test_diagnose_with_STEMI():
    filename_stemi = './tests/test_data/MI.mat'
    filename_er = './tests/test_data/BER.mat'
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

    # Fails
    filename_fail = './tests/test_data/NeurokitFails.mat'
    diagnosis_fail, explanation_fail = api.diagnose_with_STEMI(get_ecg_signal(filename_fail), sampling_rate)
    assert diagnosis_fail == Diagnosis.Failed, "Failed to handle an error during diagnostics"
    expected_explanation_fail = "Failed to diagnose due to an internal error"
    assert explanation_fail == expected_explanation_fail, f"Wrong explanation: \n\tExpected {expected_explanation_fail} \n\tGot {explanation_fail}"


def test_diagnose_with_NN_test():
    filename_not_ber = './tests/test_data/NotBER.mat'
    filename_er = './tests/test_data/BER.mat'
    filename_mi = './tests/test_data/MI.mat'
    filename_ste = './tests/test_data/STE.mat'
    filename_normal = './tests/test_data/NORMAL.mat'

    signal_not_ber = get_ecg_signal(filename_not_ber)
    signal_er = get_ecg_signal(filename_er)
    signal_mi = get_ecg_signal(filename_mi)
    signal_ste = get_ecg_signal(filename_ste)
    signal_normal = get_ecg_signal(filename_normal)

    # BER
    ber_positive = api.diagnose_early_repolarization(signal_er)
    ber_negative = api.diagnose_early_repolarization(signal_not_ber)

    assert ber_positive[0] == Diagnosis.BER, "Failed to recognize BER"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.8727"
    assert ber_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_positive[1]}"

    assert ber_negative[0] == Diagnosis.Unknown, f"Wrong explanation\n\tGot {ber_negative[0]}"
    expected_explanation = "Neutal Network calculated: the probability of BER is 0.5973"
    assert ber_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ber_negative[1]}"

    # MI
    mi_positive = api.diagnose_miocardic_infarction(signal_mi)
    mi_negative = api.diagnose_miocardic_infarction(signal_er)

    assert mi_positive[0] == Diagnosis.MI, "Failed to recognize MI"
    expected_explanation = "Neutal Network calculated: the probability of MI is 0.9953"
    assert mi_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {mi_positive[1]}"

    assert mi_negative[0] == Diagnosis.Unknown, f"Wrong explanation\n\tGot {mi_negative[0]}"
    expected_explanation = "Neutal Network calculated: the probability of MI is 0.0197"
    assert mi_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {mi_negative[1]}"

    # STE
    ste_positive = api.diagnose_ST_elevation(signal_ste)
    ste_negative = api.diagnose_ST_elevation(signal_normal)

    assert ste_positive[0] == Diagnosis.STE, "Failed to recognize STE"
    expected_explanation = "Neutal Network calculated: the probability of STE is 0.6342"
    assert ste_positive[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ste_positive[1]}"

    assert ste_negative[0] == Diagnosis.Unknown, f"Wrong explanation\n\tGot {ste_negative[0]}"
    expected_explanation = "Neutal Network calculated: the probability of STE is 0.489"
    assert ste_negative[1] == expected_explanation, f"Wrong explanation: \n\tExpected {expected_explanation} \n\tGot {ste_negative[1]}"
