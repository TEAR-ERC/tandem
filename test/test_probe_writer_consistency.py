import h5py
import pandas as pd
import numpy as np


def check_moment_rate_h5(temp_results_path, tolerances):
    """
    Checks that the moment_rate.h5 file is valid and that the sum of moment
    rates over all fault elements at the first non-zero timestep (index 0),
    matches the expected value.

    Expected: 1.2815248035796437e-06
    """
    h5_file_path = temp_results_path / "moment_rate.h5"
    assert h5_file_path.exists(), f"HDF5 file missing: {h5_file_path}"

    try:
        with h5py.File(h5_file_path, "r") as h5:
            for key in ["time", "momentRate", "faultNo", "faultVertices"]:
                assert key in h5, f"moment_rate.h5 missing dataset '{key}'"
                assert h5[key].size > 0, f"moment_rate.h5 dataset '{key}' is empty"

            time = h5["time"][:]
            moment_rate = h5["momentRate"][:]
    except OSError as e:
        raise AssertionError(
            f"moment_rate.h5 could not be opened — possibly corrupted or not "
            f"properly closed (e.g. simulation crashed mid-write): {e}"
        )

    # Validate shape: (num_fault_elements, num_timesteps, 1)
    assert moment_rate.ndim == 3, (
        f"momentRate has unexpected shape {moment_rate.shape}, "
        f"expected (fault_elements, timesteps, 1)"
    )
    assert (
        moment_rate.shape[2] == 1
    ), f"momentRate last dimension expected 1, got {moment_rate.shape[2]}"
    assert moment_rate.shape[1] == len(time), (
        f"momentRate timestep count {moment_rate.shape[1]} does not match "
        f"time array length {len(time)}"
    )

    # Sum over all fault elements at first time step
    moment_rate_sum = np.sum(moment_rate[:, 0, 0])

    expected = 1.2815248035796437e-06
    np.testing.assert_allclose(
        moment_rate_sum,
        expected,
        rtol=tolerances["seas"],
        err_msg=(
            f"Sum of moment rates at first non-zero timestep "
            f"(normalized) {moment_rate_sum} does not match expected {expected}"
        ),
    )


def check_csv_and_h5_match(prefix, temp_results_path, tolerances):
    """
    Checks that the data dumped into the CSV probe files exactly matches
    the data in the corresponding HDF5 probe files.
    """
    h5_file_path = temp_results_path / f"{prefix}.h5"
    assert h5_file_path.exists(), f"HDF5 file missing: {h5_file_path}"

    # Check HDF5 file is not corrupted or improperly closed
    # HDF5 files that crashed mid-write have a non-zero superblock status
    try:
        with h5py.File(h5_file_path, "r") as h5:
            # Check expected datasets exist and are non-empty
            for key in ["time", "probeData", "probeFields", "probeNames"]:
                assert key in h5, f"[{prefix}] HDF5 missing dataset '{key}'"
                assert h5[key].size > 0, f"[{prefix}] HDF5 dataset '{key}' is empty"

            h5_time = h5["time"][:]
            h5_data = h5["probeData"][:]
            h5_fields = [
                f.decode("ascii").strip("\x00").strip() for f in h5["probeFields"][:]
            ]
            h5_names = [
                n.decode("ascii").strip("\x00").strip() for n in h5["probeNames"][:]
            ]
    except OSError as e:
        raise AssertionError(
            f"[{prefix}] HDF5 file could not be opened — possibly corrupted or not "
            f"properly closed (e.g. simulation crashed mid-write): {e}"
        )

    # Validate shapes are consistent
    assert len(h5_names) > 0, f"[{prefix}] HDF5 contains no probes"
    assert len(h5_fields) > 0, f"[{prefix}] HDF5 contains no fields"
    assert len(h5_time) > 0, f"[{prefix}] HDF5 time array is empty"
    assert (
        h5_data.ndim == 3
    ), f"[{prefix}] probeData has unexpected shape {h5_data.shape}, expected (probes, timesteps, fields)"
    assert h5_data.shape == (len(h5_names), len(h5_time), len(h5_fields)), (
        f"[{prefix}] probeData shape {h5_data.shape} inconsistent with "
        f"probes={len(h5_names)}, timesteps={len(h5_time)}, fields={len(h5_fields)}"
    )

    for probe_idx, probe_name in enumerate(h5_names):
        csv_file_path = temp_results_path / f"{prefix}_{probe_name}.csv"
        assert csv_file_path.exists(), f"CSV file missing: {csv_file_path}"

        df = pd.read_csv(csv_file_path, comment="#")

        assert len(df) > 0, f"[{prefix}] CSV file is empty for probe {probe_name}"
        assert (
            "Time" in df.columns
        ), f"[{prefix}] CSV missing 'Time' column for probe {probe_name}"
        assert len(df) == len(h5_time), (
            f"[{prefix}] CSV row count {len(df)} does not match HDF5 timestep "
            f"count {len(h5_time)} for probe {probe_name}"
        )

        np.testing.assert_allclose(
            df["Time"].values,
            h5_time,
            rtol=tolerances["seas"],
            atol=tolerances["seas"],
            err_msg=f"[{prefix}] Time arrays do not match for probe {probe_name}",
        )

        for field_idx, field_name in enumerate(h5_fields):
            assert (
                field_name in df.columns
            ), f"[{prefix}] Field '{field_name}' missing from CSV headers"
            h5_field_data = h5_data[probe_idx, :, field_idx]
            csv_field_data = df[field_name].values

            assert not np.all(h5_field_data == 0), (
                f"[{prefix}] HDF5 field '{field_name}' is all zeros for probe "
                f"{probe_name} — possible incomplete write"
            )

            np.testing.assert_allclose(
                csv_field_data,
                h5_field_data,
                rtol=tolerances["seas"],
                atol=tolerances["seas"],
                err_msg=f"[{prefix}] Data for field '{field_name}' does not match for probe {probe_name}",
            )


def test_moment_rate(temp_results_path, tolerances):
    check_moment_rate_h5(temp_results_path, tolerances)


def test_blkst_csv_and_h5_match(temp_results_path, tolerances):
    check_csv_and_h5_match("blkst", temp_results_path, tolerances)


def test_fltst_csv_and_h5_match(temp_results_path, tolerances):
    check_csv_and_h5_match("fltst", temp_results_path, tolerances)
