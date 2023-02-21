import numpy as np


def movingAverage(ECG, W):
    new_ECG = np.zeros((ECG.shape[0], ECG.shape[1]))
    N = ECG.shape[1]
    for i in range(N):
        new_ECG[:, i] += np.mean(ECG[:, max(0, i - W // 2):min(N, i + W // 2)], axis=1)

    return new_ECG


def erosion(ECG, struct_el):
    '''
    Dilation function from signal preprocessing
    :param ECG: ECG recording of shape (n_channels, signal_length)
    :param struct_el:
        linear structuring element 2d-matrix (1d for each channel) of shape (n_channels, W)
    :return: ECG with dilation implemented
    '''
    W = struct_el.shape[1]
    assert struct_el.shape[0] == ECG.shape[0], \
        'Number of linear filters must be equal to the number of channels!'
    N = ECG.shape[1]
    new_ECG = np.zeros((ECG.shape[0], ECG.shape[1]))
    for i in range(N - W):
        to_process = ECG[:, i:(i + W)] - struct_el
        new_ECG[:, i] += np.min(to_process, axis=1)
    return new_ECG


def dilation(ECG, struct_el):
    '''
    Erosion function from signal preprocessing
    :param ECG: ECG recording of shape (n_channels, signal_length)
    :param struct_el:
        linear structuring element 2d-matrix (1d for each channel) of shape (n_channels, W)
    :return: ECG with erosion implemented
    '''
    W = struct_el.shape[1]
    assert struct_el.shape[0] == ECG.shape[0], \
        'Number of linear filters must be equal to the number of channels!'
    N = ECG.shape[1]
    new_ECG = np.zeros((ECG.shape[0], ECG.shape[1]))
    for i in range(W):
        new_ECG[:, i] += ECG[:, i] + struct_el[:, 0]

    for i in range(W, N):
        to_process = ECG[:, (i - W):i] + struct_el
        new_ECG[:, i] += np.max(to_process, axis=1)
    return new_ECG


def opening(ECG, struct_el):
    ECG = erosion(ECG, struct_el)
    ECG = dilation(ECG, struct_el)
    return ECG


def closing(ECG, struct_el):
    ECG = dilation(ECG, struct_el)
    ECG = erosion(ECG, struct_el)
    return ECG


def filter1(ECG, struct_el1, struct_el2):
    ECG = movingAverage(ECG, 15)
    ECG = movingAverage(ECG, 5)

    ECG1 = opening(ECG, struct_el1)
    ECG1 = closing(ECG1, struct_el1)

    ECG2 = closing(ECG, struct_el1)
    ECG2 = opening(ECG2, struct_el1)

    ECG3 = (ECG1 + ECG2) / 2.
    ###################################
    ECG1 = opening(ECG3, struct_el2)
    ECG1 = closing(ECG1, struct_el2)

    ECG2 = closing(ECG3, struct_el2)
    ECG2 = opening(ECG2, struct_el2)

    ECG3 = (ECG1 + ECG2) / 2.
    ##################################
    ECG = ECG - ECG3

    return ECG


def filter2(ECG, struct_el1, struct_el2):
    for i in [4, 10]:
        window = np.zeros((ECG.shape[0], i))
        ECG = (erosion(ECG, window) + dilation(ECG, window)) / 2.
    ECG = filter1(ECG, struct_el1, struct_el2)
    return ECG

def filter_ecg(ecg):
    struct1 = np.ones((ecg.shape[0], 6)) / 5
    struct2 = np.ones((ecg.shape[0], 45)) / 5
    data = filter1(ecg, struct1, struct2)
    return data
