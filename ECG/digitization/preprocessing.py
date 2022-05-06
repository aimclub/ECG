import numpy as np
import math
import cv2
from scipy import ndimage


def image_rotation(image: np.ndarray, angle: int = None) -> np.ndarray:
	if angle is None:
		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
		lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

		angles = []

		for [[x1, y1, x2, y2]] in lines:
			angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
			angles.append(angle)

		median_angle = np.median(angles)
		img_rotated = ndimage.rotate(image, median_angle)
	else:
		img_rotated = ndimage.rotate(image, angle)
	
	return img_rotated


def binarization(image: np.ndarray, threshold: float = None, inverse: bool = True) -> np.ndarray:
	assert image is not None

	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if threshold is None:
		if inverse:
			_, binaryData = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		else:
			_, binaryData = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
	else:
		if inverse:
			_, binaryData = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
		else:
			_, binaryData = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

	binary = cv2.medianBlur(binaryData, 3)
	
	return binary