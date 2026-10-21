import numpy as np

from jev_bench.report.confusion import confusion_matrix, error_gap_by_cell


def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    cm = confusion_matrix(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 0]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


def test_error_gap_attributes_extra_errors_to_cells_and_shares_sum_to_one():
    worse = np.array([[8, 2, 0], [0, 10, 0], [0, 4, 6]])  # 6 errors
    better = np.array([[9, 1, 0], [0, 10, 0], [0, 1, 9]])  # 2 errors
    gap = error_gap_by_cell(worse, better)
    assert gap["total"] == 4
    assert gap["cells"][(2, 1)] == {"extra_errors": 3, "share": 0.75}
    assert gap["cells"][(0, 1)] == {"extra_errors": 1, "share": 0.25}
    assert sum(c["share"] for c in gap["cells"].values()) == 1.0
