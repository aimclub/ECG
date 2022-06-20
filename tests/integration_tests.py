import numpy as np
import ECG.api as api
from PIL import Image
from ECG.data_classes import Diagnosis, ElevatedST, Failed, RiskMarkers,\
    TextExplanation, NNExplanation
from tests.test_util import get_ecg_signal, get_ecg_array, open_image,\
    check_data_type, compare_values
from typing import Tuple


def check_text_explanation(explanation, groundtruth_text):
    check_data_type(explanation, TextExplanation)
    compare_values(explanation.content, groundtruth_text,
                   "Unexpected explanation", multiline=True)


def check_nn_explanation(explanation, groundtruth_text):
    check_data_type(explanation, NNExplanation)
    compare_values(explanation.text, groundtruth_text,
                   "Unexpected explanation", multiline=True)
    assert len(explanation.images) > 0


def _get_NN_test_data(option):
    options = {
        'ber': './tests/test_data/BER.mat',
        'not_ber': './tests/test_data/NotBER.mat',
        'mi': './tests/test_data/MI.mat',
        'ste': './tests/test_data/STE.mat',
        'normal': './tests/test_data/NORMAL.mat'
    }
    return options[option]


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
    result = api.check_ST_elevation_with_NN(
        signal, save_path='./ECG/NN_based_approach/imgs'
    )
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], ElevatedST.Present,
                   "Failed to detect significant ST elevation")
    gt_explanation = "Significant ST elevation probability is 0.6342"
    check_nn_explanation(result[1], gt_explanation)


def test_check_ST_elevation_with_NN_absent():
    filename = _get_NN_test_data('normal')
    signal = get_ecg_signal(filename)
    result = api.check_ST_elevation_with_NN(
        signal, save_path='./ECG/NN_based_approach/imgs'
    )
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], ElevatedST.Abscent,
                   "Failed to detect absence significant ST elevation")
    gt_explanation = "Significant ST elevation probability is 0.489"
    check_nn_explanation(result[1], gt_explanation)


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
    result = api.check_BER_with_NN(signal, save_path='./ECG/NN_based_approach/imgs')
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize BER")
    gt_explanation = "BER probability is 0.8727"
    check_nn_explanation(result[1], gt_explanation)


def test_check_BER_with_NN_negative():
    filename = _get_NN_test_data('not_ber')
    signal = get_ecg_signal(filename)
    result = api.check_BER_with_NN(signal, save_path='./ECG/NN_based_approach/imgs')
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], False, "Failed to discard BER")
    gt_explanation = "BER probability is 0.5973"
    check_nn_explanation(result[1], gt_explanation)


def test_check_MI_with_NN_positive():
    filename = _get_NN_test_data('mi')
    signal = get_ecg_signal(filename)
    result = api.check_MI_with_NN(signal, save_path='./ECG/NN_based_approach/imgs')
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize MI")
    gt_explanation = "MI probability is 0.9953"
    check_nn_explanation(result[1], gt_explanation)


def test_check_MI_with_NN_negative():
    filename = _get_NN_test_data('ber')
    signal = get_ecg_signal(filename)
    result = api.check_MI_with_NN(signal, save_path='./ECG/NN_based_approach/imgs')
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], False, "Failed to discard MI")
    gt_explanation = "MI probability is 0.0197"
    check_nn_explanation(result[1], gt_explanation)


def test_check_BER_without_gradcam():
    filename = _get_NN_test_data('ber')
    signal = get_ecg_signal(filename)
    result = api.check_BER_with_NN(signal, gradcam_enabled=False)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize BER")
    assert (isinstance(result[1], NNExplanation))
    len(result[1].images) == 0


def test_check_MI_without_gradcam():
    filename = _get_NN_test_data('mi')
    signal = get_ecg_signal(filename)
    result = api.check_MI_with_NN(signal, gradcam_enabled=False)
    check_data_type(result, Tuple)
    compare_values(len(result), 2, "Wrong tuple length")
    compare_values(result[0], True, "Failed to recognize MI")
    assert(isinstance(result[1], NNExplanation))
    len(result[1].images) == 0
