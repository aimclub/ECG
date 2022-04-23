import numpy as np
from util import samples_to_ms
from math import isnan, nan

def get_intersection_for_descending_feature(ecg_cleaned, peaks, p_offsets, rr, segment_relative_length):
    segment_length = round(rr * segment_relative_length)

    intersections = []

    for i in range(len(peaks)):
        if isnan(peaks[i]) or isnan(p_offsets[i]):
            intersections.append(nan)
        else:
            baseline = ecg_cleaned[int(p_offsets[i])]

            segment_start = int(min(peaks[i] + segment_length, peaks[i]))
            segment_end = int(max(peaks[i] + segment_length, peaks[i]))

            segment = ecg_cleaned[segment_start : segment_end]

            gradient = np.gradient(segment)

            min_grad_point = [np.argmin(gradient) + segment_start, segment[np.argmin(gradient)]]
            min_grad_value = min(gradient)

            intersections.append(round((baseline - min_grad_point[1] + min_grad_value * min_grad_point[0]) / min_grad_value))

    return np.array(intersections)


def get_q_onsets(ecg_cleaned, q_peaks, p_offsets, rr, segment_relative_length=-0.15):
    return get_intersection_for_descending_feature(ecg_cleaned, q_peaks, p_offsets, rr, segment_relative_length)


def get_t_offsets(ecg_cleaned, t_peaks, p_offsets, rr, segment_relative_length=0.2):
    return get_intersection_for_descending_feature(ecg_cleaned, t_peaks, p_offsets, rr, segment_relative_length)


def get_qt_intervals(q_onsets, t_offsets):
    return t_offsets - q_onsets


def bazett(qt, rr):
    qt_sec = qt / 1000
    rr_sec = rr / 1000

    return round(qt_sec / rr_sec ** (1/2) * 1000)


def get_qtc(median_qt, rr, sampling_rate):
    median_qt_ms = samples_to_ms(median_qt, sampling_rate)
    rr_ms = samples_to_ms(rr, sampling_rate)

    return bazett(median_qt_ms, rr_ms)
