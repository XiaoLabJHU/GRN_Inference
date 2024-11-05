import sys
import os
import pytest
import numpy as np
import scipy.io

# Ensure the root directory is in the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CODE')))
from Building_MI_matrices import import_and_clean_input_file, get_dict_name, mi2_matrix_build  # noqa: E402


def test_mi2_matrix_build():
    # Define the input parameters
    input_file_name = 'tests/InSilicoSize50-Ecoli1_SS_all.tsv'
    bins_or_neighbors = 13
    mi_est = 'Shannon'

    # Load the data
    in1_data_array = import_and_clean_input_file(input_file_name)

    # Load the expected result
    dict_name = get_dict_name("MI2", bins_or_neighbors, mi_est)
    expected_result = scipy.io.loadmat('tests/test_InSilicoSize50-Ecoli1_SS_all_MI2_FB13_Shan.mat')
    expected_result = expected_result[dict_name]

    # Compute the result
    mi2_matrix_build(input_file_name, in1_data_array, bins_or_neighbors, mi_est, self_info=False)
    result = scipy.io.loadmat('tests/InSilicoSize50-Ecoli1_SS_all_MI2_FB13_Shan.mat')
    result = result[dict_name]

    # Check the result
    assert np.allclose(result, expected_result, atol=1e-10)


if __name__ == "__main__":
    pytest.main()
