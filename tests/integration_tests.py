from distutils.command.clean import clean
import math
import pickle
import numpy
import numpy as np
import ECG.api as api
from PIL import Image
from ECG.ecghealthcheck.enums import ECGClass
from ECG.data_classes import Diagnosis, ElevatedST, Failed, RiskMarkers,\
    TextExplanation, TextAndImageExplanation
from tests.test_util import get_ecg_signal, get_ecg_array, open_image,\
    check_data_type, compare_values, check_signal_shape
from typing import Tuple


def check_text_explanation(explanation, groundtruth_text):
    check_data_type(explanation, TextExplanation)
    compare_values(explanation.content, groundtruth_text,
                   "Unexpected explanation", multiline=True)


def check_text_image_explanation(explanation, groundtruth_text):
    check_data_type(explanation, TextAndImageExplanation)
    compare_values(explanation.text, groundtruth_text,
                   "Unexpected explanation", multiline=True)
    assert isinstance(explanation.image, Image.Image)


def _get_NN_test_data(option):
    options = {
        'ber': './tests/test_data/BER.mat',
        'not_ber': './tests/test_data/NotBER.mat',
        'mi': './tests/test_data/MI.mat',
        'ste': './tests/test_data/STE.mat',
        'normal': './tests/test_data/NORMAL.mat'
    }
    return options[option]


def _get_qrs_signal():
    return get_ecg_signal('./tests/test_data/NORMAL.mat')


###################
## convert image ##
###################

def test_convert_image_to_signal():
    image_filename = './tests/test_data/ecg_image.jpg'
    groundtruth_filename = './tests/test_data/ecg_array.npy'

    groundtruth = get_ecg_array(groundtruth_filename)
    image = open_image(image_filename)
    result = api.convert_image_to_signal(image)
    max_diff = np.abs(groundtruth - result).max()

    assert max_diff < 1e-14, f'Recognized signal does not match the groundtruth."\
        + "Max difference is {max_diff}'


###################
## ST-elevation ###
###################

def test_check_ST_elevation():
    filename = './tests/test_data/MI.mat'
    signal = get_ecg_signal(filename)
    result = api.check_ST_elevation(signal, sampling_rate=500)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], ElevatedST.Present,
                   "Failed to detect significant ST elevation")
    gt_explanation = "ST elevation value in lead V3 (0.225 mV) exceeded "\
        + "the threshold 0.2, therefore ST elevation was detected."
    check_text_explanation(result[1], gt_explanation)


def test_check_ST_elevation_failure():
    filename = './tests/test_data/NeurokitFails.mat'
    result = api.check_ST_elevation(get_ecg_signal(filename), sampling_rate=500)
    check_data_type(result, Failed)


def test_check_ST_elevation_with_NN_present():
    filename = _get_NN_test_data('ste')
    signal = get_ecg_signal(filename)
    result = api.check_ST_elevation_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], ElevatedST.Present,
                   "Failed to detect significant ST elevation")
    gt_explanation = "Significant ST elevation probability is 0.6342"
    check_text_image_explanation(result[1], gt_explanation)


def test_check_ST_elevation_with_NN_absent():
    filename = _get_NN_test_data('normal')
    signal = get_ecg_signal(filename)
    result = api.check_ST_elevation_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], ElevatedST.Abscent,
                   "Failed to detect absence significant ST elevation")
    gt_explanation = "Significant ST elevation probability is 0.489"
    check_text_image_explanation(result[1], gt_explanation)


###################
## risk markers ###
###################

def test_evaluate_risk_markers():
    filename = './tests/test_data/MI.mat'
    signal = get_ecg_signal(filename)
    result = api.evaluate_risk_markers(signal, sampling_rate=500)
    check_data_type(result, RiskMarkers)
    compare_values(result.Ste60_V3, 0.225, "Failed to evaluate STE60 V3")
    compare_values(result.QTc, 501, "Failed to evaluate QTc")
    compare_values(result.RA_V4, 0.315, "Failed to evaluate RA V4")


def test_evaluate_risk_markers_failure():
    filename = './tests/test_data/NeurokitFails.mat'
    result = api.evaluate_risk_markers(get_ecg_signal(filename), sampling_rate=500)
    check_data_type(result, Failed)


###################
## diagnose #######
###################

def test_diagnose_with_risk_markers_MI():
    filename = './tests/test_data/MI.mat'
    signal = get_ecg_signal(filename)
    result = api.diagnose_with_risk_markers(signal, sampling_rate=500, tuned=False)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], Diagnosis.MI, "Failed to recognize MI")
    gt_explanation = "Criterion value calculated as follows: "\
        + "(1.196 * [STE60 V3 in mm]) + (0.059 * [QTc in ms])"\
        + " – (0.326 * [RA V4 in mm])) = 31.2231 exceeded the threshold 23.4,"\
        + " therefore the diagnosis is Myocardial Infarction"
    check_text_explanation(result[1], gt_explanation)


def test_diagnose_with_risk_markers_MI_tuned():
    filename = './tests/test_data/MI.mat'
    signal = get_ecg_signal(filename)
    result = api.diagnose_with_risk_markers(signal, sampling_rate=500, tuned=True)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], Diagnosis.MI, "Failed to recognize MI")
    check_data_type(result[1], TextExplanation)
    gt_explanation = "Criterion value calculated as follows: "\
        + "(2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms])"\
        + " + (-1.7 * np.minimum([RA V4 in mm], 19)) = 151.47 "\
        + "exceeded the threshold 126.9,"\
        + " therefore the diagnosis is Myocardial Infarction"
    check_text_explanation(result[1], gt_explanation)


def test_diagnose_with_risk_markers_BER_tuned():
    filename = './tests/test_data/BER.mat'
    signal = get_ecg_signal(filename)
    result = api.diagnose_with_risk_markers(signal, sampling_rate=500, tuned=True)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], Diagnosis.BER, "Failed to recognize BER")
    gt_explanation = "Criterion value calculated as follows: "\
        + "(2.9 * [STE60 V3 in mm]) + (0.3 * [QTc in ms])"\
        + " + (-1.7 * np.minimum([RA V4 in mm], 19)) = 118.4062869471591"\
        + " did not exceed the threshold 126.9,"\
        + " therefore the diagnosis is Benign Early Repolarization"
    check_text_explanation(result[1], gt_explanation)


def test_check_BER_with_NN_positive():
    filename = _get_NN_test_data('ber')
    signal = get_ecg_signal(filename)
    result = api.check_BER_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize BER")
    gt_explanation = "BER probability is 0.8727"
    check_text_image_explanation(result[1], gt_explanation)


def test_check_BER_with_NN_negative():
    filename = _get_NN_test_data('not_ber')
    signal = get_ecg_signal(filename)
    result = api.check_BER_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], False, "Failed to discard BER")
    gt_explanation = "BER probability is 0.5973"
    check_text_image_explanation(result[1], gt_explanation)


def test_check_MI_with_NN_positive():
    filename = _get_NN_test_data('mi')
    signal = get_ecg_signal(filename)
    result = api.check_MI_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize MI")
    gt_explanation = "MI probability is 0.9953"
    check_text_image_explanation(result[1], gt_explanation)


def test_check_MI_with_NN_negative():
    filename = _get_NN_test_data('ber')
    signal = get_ecg_signal(filename)
    result = api.check_MI_with_NN(signal)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], False, "Failed to discard MI")
    gt_explanation = "MI probability is 0.0197"
    check_text_image_explanation(result[1], gt_explanation)


def test_check_ecg_is_normal_positive():
    filename = './tests/test_data/class_norm.mat'
    signal = get_ecg_signal(filename, read_nested=False)[:, :4000]
    check_signal_shape(signal.shape, (12, 4000), "Wrong signal shape")
    result = api.check_ecg_is_normal(signal, ECGClass.ALL)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to classify signal")
    check_text_explanation(result[1], "The signal is ok")


def test_check_ecg_is_normal_negative():
    filename = './tests/test_data/class_abnorm.mat'
    signal = get_ecg_signal(filename, read_nested=False)[:, :4000]
    check_signal_shape(signal.shape, (12, 4000), "Wrong signal shape")
    result = api.check_ecg_is_normal(signal, ECGClass.ALL)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], False, "Failed to classify signal")
    check_text_explanation(result[1], "The signal has some abnormalities")


###################
## QRS complex ####
###################

def _is_equal_peak_value(observed_value, expected_value):
    if math.isnan(expected_value):
        return math.isnan(observed_value)
    else:
        return observed_value == expected_value


def _compare_peak_values(observed, expected, message):
    check_data_type(observed, list, message)
    compare_values(len(observed), len(expected), f'{message}, checking length')
    values_equality = [_is_equal_peak_value(o, e) for (o, e) in zip(observed, expected)]
    assert np.array(values_equality).all(), \
        f'{message}. Wrong values: expected {expected}, got {observed}'


def test_get_qrs_complex_success():
    signal = _get_qrs_signal()
    sampling_rate = 500
    result = api.get_qrs_complex(signal, sampling_rate)

    check_data_type(result, tuple)
    assert len(result) == 2, f"expected tuple of length 2, got {len(result)}"
    cleaned_signal, peaks = result

    # check signal
    check_data_type(cleaned_signal, numpy.ndarray)
    expected_cleaned_signal = np.load('./tests/test_data/cleaned.npy')
    assert cleaned_signal.shape == expected_cleaned_signal.shape, \
        f'expected array of shape {expected_cleaned_signal.shape}, ' \
        + f'got array of shape {cleaned_signal.shape}'
    assert (cleaned_signal == expected_cleaned_signal).all(), \
        'cleaned signal does not match expected value'

    # check peaks
    check_data_type(peaks, list)
    for i, el in enumerate(peaks):
        assert set(el.keys()) == set('PQRST'), f'wrong keys in {i}-th element'
    with open('tests/test_data/peaks.pkl', 'rb') as f:
        expected_peaks = pickle.load(f)
    for i in range(len(expected_peaks)):
        for key in 'PQRST':
            _compare_peak_values(
                peaks[i][key],
                expected_peaks[i][key],
                f'Channel {i}, wave {key}'
            )


def test_get_qrs_complex_fail():
    signal = _get_qrs_signal()
    sampling_rate = 500
    result = api.get_qrs_complex(signal[0], sampling_rate)
    check_data_type(result, Failed)
