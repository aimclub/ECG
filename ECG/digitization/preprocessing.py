import numpy as np
import math
import cv2
from scipy import ndimage


def image_rotation(image: np.ndarray, angle: int = None) -> np.ndarray:
    if angle is None:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 15,
                                minLineLength=40, maxLineGap=5)

        angles = []

        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)
        img_rotated = ndimage.rotate(image, median_angle)
    else:
        img_rotated = ndimage.rotate(image, angle)

    return img_rotated


def automatic_brightness_and_contrast(image: np.ndarray,
                                      clip_hist_percent: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result


def shadow_remove(image: np.ndarray) -> np.ndarray:
    rgb_planes = cv2.split(image)
    result_norm_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        shadow_remov = cv2.merge(result_norm_planes)

    return shadow_remov


def warming_filter(image: np.ndarray) -> np.ndarray:
    originalValues = np.array([0, 50, 100, 150, 200, 255])
    redValues = np.array([0, 80, 150, 190, 220, 255])
    blueValues = np.array([0, 20, 40, 75, 150, 255])

    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(image)

    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)

    result = cv2.merge([B, G, R])

    return result


def adjust_image(image: np.ndarray) -> np.ndarray:
    auto_bc_image = automatic_brightness_and_contrast(image)
    adjusted_image = shadow_remove(auto_bc_image)
    warm_image = warming_filter(adjusted_image)
    rotated_image = image_rotation(warm_image)

    return rotated_image


def binarization(image: np.ndarray, threshold: float = None,
                 inverse: bool = True) -> np.ndarray:
    assert image is not None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if threshold is None:
        if inverse:
            _, binaryData = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binaryData = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    else:
        if inverse:
            _, binaryData = cv2.threshold(
                grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binaryData = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binaryData, 3)

    return binary
