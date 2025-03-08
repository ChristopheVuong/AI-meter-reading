import pytest
from typing import List
import pandas as pd
from utils.metrics import accuracyModulo1000


@pytest.mark.parametrize(
    "test_labels_df, true_labels_df, expected",
    [
        # Case 1: Perfect accuracy
        (
            pd.DataFrame({"filename": ["au.jpg", "bu.jpg"], "index": ["456", "671"]}),
            pd.DataFrame({"filename": ["au.jpg", "bu.jpg"], "index": ["456", "671"]}),
            1.0,
        ),
        # Case 2: Partial accuracy
        (
            pd.DataFrame({"filename": ["cu.jpg", "du.jpg"], "index": ["123", "789"]}),
            pd.DataFrame({"filename": ["cu.jpg", "du.jpg"], "index": ["123", "456"]}),
            0.5,
        ),
        # Case 3: No correct predictions
        (
            pd.DataFrame({"filename": ["eu.jpg", "fu.jpg"], "index": ["111", "222"]}),
            pd.DataFrame({"filename": ["gu.jpg", "hu.jpg"], "index": ["333", "444"]}),
            0.0,
        ),
        # Case 4: Empty inputs
        (
            pd.DataFrame(columns=["filename", "index"]),
            pd.DataFrame(columns=["filename", "index"]),
            0.0,
        ),
        # Case 5: Single element, correct prediction
        (
            pd.DataFrame({"filename": ["single.jpg"], "index": ["1"]}),
            pd.DataFrame({"filename": ["single.jpg"], "index": ["1"]}),
            1.0,
        ),
        # Case 6: Single element, incorrect prediction
        (
            pd.DataFrame({"filename": ["single.jpg"], "index": ["1"]}),
            pd.DataFrame({"filename": ["other.jpg"], "index": ["2"]}),
            0.0,
        ),
    ],
)
def test_accuracyModulo1000(
    test_labels_df: pd.DataFrame, true_labels_df: pd.DataFrame, expected: float
):
    """
    Test accuracy modulo 1000 calculation using pytest's parametrize feature.
    """
    # Call the function under test
    accuracy: float = accuracyModulo1000(test_labels_df, true_labels_df)

    # Assert that the result matches the expected value
    assert accuracy == pytest.approx(expected), (
        f"Test failed for predictions={test_labels_df}, labels={true_labels_df}. "
        f"Expected {expected}, but got {accuracy}"
    )
