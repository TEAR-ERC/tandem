import pandas as pd
import numpy as np


def detect_peaks(file_name, window_size=1e9, relative_error=0.5):
    df = pd.read_csv(file_name, comment="#")
    time = df["Time"].values
    vmax = df["VMax"].values

    peaks = []
    i = 0
    n = len(time)

    while i < n:
        window_start = time[i]
        window_end = window_start + window_size

        # Find the indices within the window
        window_mask = (time >= window_start) & (time <= window_end)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) < 3:
            i += 1
            continue

        v_window = vmax[window_indices]
        t_window = time[window_indices]

        v_max = np.max(v_window)
        idx_max = np.argmax(v_window)
        t_max = t_window[idx_max]

        v_left = v_window[0]
        v_right = v_window[-1]

        # Check if the max is significantly higher than both edges
        left_check = abs(v_max - v_left) / v_max >= relative_error
        right_check = abs(v_max - v_right) / v_max >= relative_error

        if left_check and right_check:
            peaks.append((t_max, v_max))

            # Move to end of window to avoid overlapping detections
            i = window_indices[-1] + 1
        else:
            i += 1
    return peaks


def check_SEAS_consistency_2D(file_vmax, file_vmax_gf):
    window_size = 1e9
    relative_error = 0.5

    peaks = detect_peaks(file_vmax, window_size, relative_error)
    peaks_gf = detect_peaks(file_vmax_gf, window_size, relative_error)

    assert len(peaks) == len(
        peaks_gf
    ), "Number of peaks detected does not match between files."

    peak_time_interval_QD = [
        peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)
    ]
    peak_time_intervals_QDGreen = [
        peaks_gf[i + 1][0] - peaks_gf[i][0] for i in range(len(peaks_gf) - 1)
    ]
    print(peak_time_interval_QD)
    print(peak_time_intervals_QDGreen)
    assert np.allclose(
        peak_time_interval_QD, peak_time_intervals_QDGreen, rtol=1e-8
    ), "Peak time intervals do not match between files."


def test_SEAS_consistency_QD_2D(results_path):
    file_vmax_ref = results_path / "vmax_ref_QD.csv"
    file_vmax_output = results_path / "vmax_output_QD.csv"
    check_SEAS_consistency_2D(file_vmax_ref, file_vmax_output)


def test_SEAS_consistency_QDGreen_2D(results_path):
    file_vmax_ref = results_path / "vmax_ref_QDGreen.csv"
    file_vmax_output = results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency_2D(file_vmax_ref, file_vmax_output)


def test_SEAS_consistency_QD_vs_QDGreen_2D(results_path):
    file_vmax = results_path / "vmax_output_QD.csv"
    file_vmax_gf = results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency_2D(file_vmax, file_vmax_gf)
