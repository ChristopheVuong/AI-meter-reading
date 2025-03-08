import pandas as pd
import numpy as np


def joinCompDf(df: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames (`df` and `df_ref`) on the 'filename' column and ensure
    specific columns ('index' and 'index_pred') are properly formatted as integers.

    Parameters:
        df (pd.DataFrame): The first DataFrame containing prediction data (string).
        df_ref (pd.DataFrame): The reference DataFrame containing ground truth (string).

    Returns:
        pd.DataFrame: A merged DataFrame with 'index' and 'index_pred' columns formatted as integers.
    """
    joint_labels_df = pd.merge(df_ref, df, on="filename", how="inner")
    # check if the index_pred is a column name of test_labels_df otherwise exceptions
    if (
        "index" not in joint_labels_df.columns
        or "index_pred" not in joint_labels_df.columns
    ):
        raise ValueError(
            "Both 'index' and 'index_pred' columns must be present in the merged dataframe."
        )
    # ensure that the fields are ready as type integer
    joint_labels_df["index"] = joint_labels_df["index"].astype(int)
    joint_labels_df["index_pred"] = (
        joint_labels_df["index_pred"].astype(int).apply(np.ceil)
    )  # adapted to regression float outputs
    return joint_labels_df


def accuracyModulo1000(
    test_labels_df: pd.DataFrame, true_labels_df: pd.DataFrame
) -> float:
    """
    Calculate the accuracy of predictions when comparing index values modulo 1000.

    Parameters:
        test_labels_df (pd.DataFrame): DataFrame containing predicted indices in 'index_pred' column.
        true_labels_df (pd.DataFrame): DataFrame containing true indices in 'index' column.

    Returns:
        float: The proportion of cases where (index % 1000) matches (index_pred % 1000).
    """
    joint_labels_df = joinCompDf(test_labels_df, true_labels_df)

    accuracy = (
        joint_labels_df["index"] % 1000 == joint_labels_df["index_pred"] % 1000
    ).mean()
    return accuracy


# introduce metrics Levenhstein, edit distance (treat the indices as sequences)
# test we tolerate an index that is longer by one or two characters assuming that the index is contained in the string
# why because we can still have a good prediction if the index is contained in the string how can we guess the right
# measure, by looking at the usual highest units of the consumer in history and the number of digits in the index.
# But such weird measure should be highlighted and by cross examination, the rate of success should be higher.


def strStrRabinKarpModulo(haystack: str, needle: str) -> int:
    """
    :type haystack: str
    :type needle: str
    :return type: int
    Rabin-Karp Substring Search: One major advantage of Rabin-Karp is that it uses O(1) auxiliary storage space,
    which is great if the pattern string you're looking for is very large.
    Pre-compute efficiently a hash map for each sequence of len(needle) characters within haystack, we can find the locations
    of needle in O(len(haystack)) time.
    Very fast in practice because of modulo (beats 70%)
    Rabin fingerprint: string as a base 128 (or however many characters are in our alphabet) number
    complexity O(len(needle))

    Modify it to rule out sequences that are different in length by more than OFFSET_TOL characters
    """
    ### VALUES FOR RABIN-KARP algorithm
    MAX_CHAR = 256  # 128 or 256
    P = 101 # 19 for MAX_CHAR = 26
    OFFSET_TOL = 2

    ### END VALUES FOR RABIN-KARP algorithm
    def hashcodeModulo(s: str, start: int, end: int) -> int:
        hash = 0
        for i in range(start, end):
            hash = (
                hash * MAX_CHAR + ord(s[i])
            ) % P  # so that the contribution of each character is less than MAX_CHAR
        return hash

    # Should consider MAX_CHAR = 256 to cover all ASCII characters and p = 101 modulo for example (more characters)
    n = len(needle)
    m = len(haystack)
    h = 1
    if n == 0:
        return 0
    if m < n or abs(m - n) > OFFSET_TOL:
        return -1

    # Pre-compute the highest power of MAX_CHAR modulo P h = 1
    for i in range(n - 1):
        h = (h * MAX_CHAR) % P

    hashcodeNeedle = hashcodeModulo(needle, 0, n)
    hashcodeSubstring = hashcodeModulo(haystack, 0, n)
    for i in range(m - n + 1):
        if (
            hashcodeSubstring == hashcodeNeedle and haystack[i : i + n] == needle
        ):  # handle collision
            return i
        if i < m - n:
            # shift to the right removing the current leftmost character and add the hashcode associated to the next one
            hashcodeSubstring = (
                MAX_CHAR * (hashcodeSubstring - h * ord(haystack[i]))
                + ord(haystack[i + n])
            ) % P
            # shift by 1 to the right (rolling hash)
            if hashcodeSubstring < 0:
                hashcodeSubstring += P  # keep the modulo remainder positive
    return -1


# introduce a metrics for digit precision based on edit distance for each sequences with same length
# this highlight how precise an individual prediction is in terms of the digits in average
def digit_precision(
    test_labels_df: pd.DataFrame, true_labels_df: pd.DataFrame
) -> float:
    """
    Compare two pandas dataframe corresponding to test and true labels with the following columns respectively:
    - index_pred
    - index
    """
    joint_labels_df = joinCompDf(test_labels_df, true_labels_df)

    true_positives = (
        joint_labels_df["index"] % 1000 == joint_labels_df["index_pred"] % 1000
    ).sum()
    false_positives = (
        joint_labels_df["index"] % 1000 != joint_labels_df["index_pred"] % 1000
    ).sum()
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    return precision


# introduce a last metric sort of digit_recall : the rate of correct digits in the predictions (sum)
# this highlight the quality as an OCR for digits (regardless of the lengths of the sequences)


def digit_recall(test_labels_df: pd.DataFrame, true_labels_df: pd.DataFrame) -> float:
    pass
