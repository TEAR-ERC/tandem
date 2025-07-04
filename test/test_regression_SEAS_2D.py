import pandas as pd
import numpy as np


def detect_events(file_name, window_size=1e9, relative_error=0.5):
    df = pd.read_csv(file_name, comment="#")
    time = df["Time"].values
    vmax = df["VMax"].values

    events = []
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
            events.append((t_max, v_max))

            # Move to end of window to avoid overlapping detections
            i = window_indices[-1] + 1
        else:
            i += 1
    return events


def check_SEAS_consistency_2D(
    file_vmax, file_vmax_gf, tolerance_seas, tolerance_seas_events
):
    window_size = 1e9  # seconds
    relative_error = 0.5  # the max in a window_size window has to be relatively 50% above the values at the two edges of the window to be considered an event

    events = detect_events(file_vmax, window_size, relative_error)
    events_gf = detect_events(file_vmax_gf, window_size, relative_error)

    sorted_events = sorted(events)
    sorted_events_gf = sorted(events_gf)

    arr1 = np.array(sorted_events)
    arr2 = np.array(sorted_events_gf)

    times1 = arr1[:, 0]
    times2 = arr2[:, 0]

    values1 = arr1[:, 1]
    values2 = arr2[:, 1]

    assert np.allclose(times1, times2, rtol=tolerance_seas), "Event times do not match"
    assert np.allclose(
        values1, values2, rtol=tolerance_seas_events
    ), "Event slip rate magnitudes do not match"

    event_time_interval_QD = [
        events[i + 1][0] - events[i][0] for i in range(len(events) - 1)
    ]
    event_time_intervals_QDGreen = [
        events_gf[i + 1][0] - events_gf[i][0] for i in range(len(events_gf) - 1)
    ]
    assert np.allclose(
        event_time_interval_QD,
        event_time_intervals_QDGreen,
        rtol=tolerance_seas,
    ), "event time intervals do not match between files."


def test_SEAS_consistency_QD_2D(
    temp_results_path, reference_results_path, tolerance_seas, tolerance_seas_events
):
    file_vmax_ref = reference_results_path / "vmax_ref_QD.csv"
    file_vmax_output = temp_results_path / "vmax_output_QD.csv"
    check_SEAS_consistency_2D(
        file_vmax_ref, file_vmax_output, tolerance_seas, tolerance_seas_events
    )


def test_SEAS_consistency_QDGreen_2D(
    temp_results_path, reference_results_path, tolerance_seas, tolerance_seas_events
):
    file_vmax_ref = reference_results_path / "vmax_ref_QDGreen.csv"
    file_vmax_output = temp_results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency_2D(
        file_vmax_ref, file_vmax_output, tolerance_seas, tolerance_seas_events
    )


def test_SEAS_consistency_QD_vs_QDGreen_2D(
    temp_results_path, tolerance_seas, tolerance_seas_events
):
    file_vmax = temp_results_path / "vmax_output_QD.csv"
    file_vmax_gf = temp_results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency_2D(
        file_vmax, file_vmax_gf, tolerance_seas, tolerance_seas_events
    )
