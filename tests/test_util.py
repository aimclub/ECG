import scipy.io
import numpy as np
from pathlib import Path
from PIL import Image


def get_ecg_signal(filename, read_nested=True):
    mat = scipy.io.loadmat(filename)

    return np.array(mat['ECG'][0][0][2]) if read_nested else mat['ECG']


def get_ecg_array(filename):
    data = np.load(filename)

    return data


def open_image(path):
    assert Path(path).exists()

    data = Image.open(path)
    assert data is not None

    return data


def _check_data_type_message(object, expected_type, message):
    if message is None:
        return f"Wrong data type: expected {expected_type}, got {type(object)}"
    else:
        return f"{message}. Wrong data type: expected {expected_type}, got {type(object)}"


def check_data_type(object, expected_type, message=None):
    assert isinstance(object, expected_type), \
        _check_data_type_message(object, expected_type, message)


def compare_values(value, groundtruth, message, multiline=False):
    sep = '\n\t' if multiline else ''
    assert groundtruth == value, \
        f'{message}. {sep}Expected {groundtruth}. {sep}Got {value}.'


def check_signal_shape(shape, expected_shape, message):
    assert shape == expected_shape, \
        f'{message}. Expected shape {expected_shape}/ Got {shape}.'
