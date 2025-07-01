import os
import numpy as np
import vtk
import glob
import pytest
from pathlib import Path
import pandas as pd


@pytest.fixture(scope="module")
def tolerance():
    return 1e-10


@pytest.fixture(scope="module")
def results_path():
    return Path("/app/test/test_data/reference_results")
