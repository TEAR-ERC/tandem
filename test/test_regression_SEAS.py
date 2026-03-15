import pandas as pd
import numpy as np


def detect_events(file_name, window_size=1e9, relative_error=0.5):
    """
    Detect events from a CSV file containing 'Time' and 'VMax' columns.

    An event is defined as a local maximum that is significantly above the
    values at the edges of a time window (controlled by relative_error).
    Returns a list of (time, vmax) events.
    """
    df = pd.read_csv(file_name, comment="#")
    time = df["Time"].values
    vmax = df["VMax"].values

    assert np.all(np.diff(time) >= 0), "Time array must be sorted for event detection"

    events = []
    i = 0
    n = len(time)

    while i < n:
        window_start = time[i]
        window_end = window_start + window_size

        end_idx = np.searchsorted(time, window_end, side="right")

        v_window = vmax[i:end_idx]
        t_window = time[i:end_idx]

        if len(v_window) < 3:
            break  # not enough points left, no more events possible

        v_max = np.max(v_window)
        t_max = t_window[np.argmax(v_window)]

        v_left = v_window[0]
        v_right = v_window[-1]

        # Check if the max is significantly higher than both edges
        left_check = abs(v_max - v_left) / v_max >= relative_error
        right_check = abs(v_max - v_right) / v_max >= relative_error

        if left_check and right_check:
            events.append((t_max, v_max))
            i = end_idx
        else:
            i += 1
    return events


def check_SEAS_consistency(file_reference, file_tested, tolerances):
    """
    Compare event detections between a reference and tested CSV file.

    Asserts that the number, times, and magnitudes of detected events match
    within provided tolerances.
    """
    window_size = 1e9  # seconds
    relative_error = 0.5  # the max in a window_size window has to be relatively 50% above the values at the two edges of the window to be considered an event

    events_reference = detect_events(file_reference, window_size, relative_error)
    events_tested = detect_events(file_tested, window_size, relative_error)

    assert len(events_reference) == len(events_tested), (
        f"Number of detected events does not match: "
        f"{len(events_reference)} in reference vs {len(events_tested)} in tested"
    )

    arr_ref = np.array(sorted(events_reference))
    arr_tst = np.array(sorted(events_tested))

    times_reference = arr_ref[:, 0]
    times_tested = arr_tst[:, 0]

    values_reference = arr_ref[:, 1]
    values_tested = arr_tst[:, 1]

    assert np.allclose(
        times_reference, times_tested, rtol=tolerances["seas"]
    ), "Event times do not match"
    assert np.allclose(
        values_reference, values_tested, rtol=tolerances["seas_events"]
    ), "Event slip rate magnitudes do not match"

    event_time_intervals_reference = [
        events_reference[i + 1][0] - events_reference[i][0]
        for i in range(len(events_reference) - 1)
    ]
    event_time_intervals_tested = [
        events_tested[i + 1][0] - events_tested[i][0]
        for i in range(len(events_tested) - 1)
    ]
    assert np.allclose(
        event_time_intervals_reference,
        event_time_intervals_tested,
        rtol=tolerances["seas"],
    ), "Event time intervals do not match between files."


def test_SEAS_consistency_QD(temp_results_path, reference_results_path, tolerances):
    """Regression test comparing QD reference and test VMax CSV outputs."""
    file_vmax_ref = reference_results_path / "vmax_ref_QD.csv"
    file_vmax_output = temp_results_path / "vmax_output_QD.csv"
    check_SEAS_consistency(file_vmax_ref, file_vmax_output, tolerances)


def test_SEAS_consistency_QDGreen(
    temp_results_path, reference_results_path, tolerances
):
    """Regression test comparing QDGreen reference and test VMax CSV outputs."""
    file_vmax_ref = reference_results_path / "vmax_ref_QDGreen.csv"
    file_vmax_output = temp_results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency(file_vmax_ref, file_vmax_output, tolerances)


def test_SEAS_consistency_QD_vs_QDGreen(temp_results_path, tolerances):
    """
    Compare QD and QDGreen outputs from the same run to ensure they
    produce consistent event detection results.
    """
    file_vmax = temp_results_path / "vmax_output_QD.csv"
    file_vmax_gf = temp_results_path / "vmax_output_QDGreen.csv"
    check_SEAS_consistency(file_vmax, file_vmax_gf, tolerances)
