from ECG.condition.enums import ECGClass

few_shot_files = {
    ECGClass.NORM: [
        '01630.mat',
        '19669.mat',
        '01700.mat',
        '15090.mat',
        '12886.mat',
        '09449.mat',
        '05946.mat'],
    ECGClass.ALL: [
        '07003.mat',
        '17325.mat',
        '02119.mat',
        '20313.mat',
        '04476.mat',
        '12646.mat',
        '08185.mat'],
    ECGClass.STTC: [
        '17309.mat',
        '10115.mat',
        '02322.mat',
        '00569.mat',
        '04508.mat',
        '08139.mat',
        '18520.mat'],
    ECGClass.MI: [
        '16712.mat',
        '10837.mat',
        '03052.mat',
        '00947.mat',
        '04646.mat',
        '08278.mat',
        '17951.mat']}

normalization_params = {
    'mean': [
        8.84816248e-07,
        5.44832773e-06,
        4.56945345e-06,
        -3.14865769e-06,
        -1.93700007e-06,
        4.96820982e-06,
        1.35710232e-05,
        -5.82670901e-06,
        6.16073251e-06,
        1.02679495e-05,
        1.69133085e-05,
        -7.29526504e-05],
    'std': [
        0.13267802,
        0.13312846,
        0.13415672,
        0.11368822,
        0.11456738,
        0.11643286,
        0.18899392,
        0.29433406,
        0.28234617,
        0.24142601,
        0.21591534,
        0.17862277]}

ECG_LENGTH = 4000

FILTER_METHOD = 'neurokit'
