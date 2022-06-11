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