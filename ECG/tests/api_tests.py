import ECG.api as api
from ECG.tests.test_util import get_ecg_signal

def convert_image_to_signal_test():
    raise NotImplementedError()


def check_ST_test():
    filename = './ECG/tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename)

    ste = api.check_ST_elevation(signal, sampling_rate)
    
    print('\n***')
    print('check_ST_test\n')
    print('Expected: 0.225')
    print('Got:      ' + str(ste))
    print('***\n')


def evaluate_risk_markers_test():
    filename = './ECG/tests/test_data/MI.mat'
    sampling_rate = 500
    signal = get_ecg_signal(filename)

    risk_markers = api.evaluate_risk_markers(signal, sampling_rate)
    
    print('\n***')
    print('evaluate_risk_markers_test\n')

    print('STE60 V3')
    print('Expected: 0.225')
    print('Got:      ' + str(risk_markers.Ste60_V3) + '\n')

    print('QTc')
    print('Expected: 501')
    print('Got:      ' + str(risk_markers.QTc) + '\n')

    print('RA V4')
    print('Expected: 0.315')
    print('Got:      ' + str(risk_markers.RA_V4))
    print('***\n')


def diagnose_with_STEMI_test():
    filename_stemi = './ECG/tests/test_data/MI.mat'
    filename_er = './ECG/tests/test_data/BER.mat'
    sampling_rate = 500
    signal_stemi = get_ecg_signal(filename_stemi)
    signal_er = get_ecg_signal(filename_er)

    stemi_positive = api.diagnose_with_STEMI(signal_stemi, sampling_rate)
    stemi_negative = api.diagnose_with_STEMI(signal_er, sampling_rate)
    
    print('\n***')
    print('diagnose_with_STEMI_test\n')
    print('Diagnosis')
    print('Expected: Myocardial Infarction')
    print('Got:      ' + str(stemi_positive[0].value) + '\n')
    print('Expected: Criterion value calculated as follows: (1.0 * [STE60 V3 in mm]) + (0.06 * [QTc in ms]) - (0.5 * min([RA V4 in mm], 10.5)) = 30.735000000000003 did not exceed the threshold 27.1, therefore the diagnosis is Myocardial Infarction')
    print('Got:      ' + str(stemi_positive[1]) + '\n')

    print('Diagnosis')
    print('Expected: Benign Early Repolarization')
    print('Got:      ' + str(stemi_negative[0].value) + '\n')
    print('Expected: Criterion value calculated as follows: (1.0 * [STE60 V3 in mm]) + (0.06 * [QTc in ms]) - (0.5 * min([RA V4 in mm], 10.5)) = 25.94940929212383 exceeded the threshold 27.1, therefore the diagnosis is Benign Early Repolarization')
    print('Got:      ' + str(stemi_negative[1]) + '\n')

    print('***\n')
