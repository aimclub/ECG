import numpy as np
import cv2
from scipy import signal
import PIL
from PIL import Image
import warnings


def find_interval(gap: np.ndarray) -> float:
    new_arr = np.diff(gap)
    result = np.delete(new_arr, np.where(new_arr == 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan if np.isnan(np.mean(result)) else np.mean(result)


def resize_pic(image: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    resize_coef = 2000 / image.size[0]
    pic_h = int(image.size[1] * resize_coef)
    resized_im = image.resize((2000, pic_h), PIL.Image.NEAREST)
    resized_im = np.asarray(resized_im)

    return resized_im


def grid_detection(image: np.ndarray) -> float:
    assert image is not None

    if image.shape[1] < 2000:
        img_w = image.shape[1]
        kf = int(np.ceil(img_w * 15 / 2000))
        kernel_size = kf + 1 if kf % 2 == 0 else kf
        c_kf = int(np.ceil(img_w * 6 / 2000))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
        grid = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, c_kf)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = grayscale.astype('float32')
        intensity_shift = 20
        grayscale += intensity_shift
        grayscale = np.clip(grayscale, 0, 255)
        grayscale = grayscale.astype('uint8')
        blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
        grid = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 6)
        grid = cv2.medianBlur(grid, 3)

    intervals = []
    for n_row in range(grid.shape[0]):
        intervals.append(find_interval(np.where(grid[n_row, :] == 0)[0]))
    for n_col in range(grid.shape[1]):
        intervals.append(find_interval(np.where(grid[:, n_col] == 0)[0]))

    scale = np.nanmean(np.array(intervals))
    scale = scale.round(1)

    return scale


def signal_extraction(image: np.ndarray, scale: float) -> float:
    assert image is not None

    ecg = image
    max_value = 0
    x_row = 0
    for row in range(len(ecg)):
        if ecg[row].sum() > max_value:
            max_value = ecg[row].sum()
            x_row = row

    Y = []
    for col in range(len(ecg[1])):
        for row in range(len(ecg[:, col].flatten())):
            if ecg[:, col].flatten()[row] == 255:
                y = (x_row - row) / scale
                Y.append(y)
                break

    sig = Y
    win = signal.windows.hann(10)
    filtered = signal.convolve(sig, win, mode='same') / sum(win)

    return filtered
