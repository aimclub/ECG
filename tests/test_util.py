import scipy.io
import numpy as np
from pathlib import Path
from PIL import Image


def get_ecg_signal(filename):
    mat = scipy.io.loadmat(filename)

    return np.array(mat['ECG'][0][0][2])


def get_ecg_array(filename):
    data = np.load(filename)

    return data


def open_image(path):
    assert Path(path).exists()

    data = Image.open(path)
    assert data is not None

    return data


def check_data_type(object, expected_type):
    assert isinstance(object, expected_type), \
        f"Wrong data type: expected {expected_type}, got {type(object)}"


def compare_values(value, groundtruth, message, multiline=False):
    sep = '\n\t' if multiline else ''
    assert groundtruth == value, \
        f'{message}. {sep}Expected {groundtruth}. {sep}Got {value}.'
