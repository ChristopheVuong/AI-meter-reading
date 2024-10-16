import pandas as pd
import numpy as np

def accuracyModulo1000(test_labels, true_labels):
    """
    Compare two pandas dataframe corresponding to test and true labels
    """
    joint_labels_df = pd.merge(labels_df, labels_predicted_df, on='filename', how='inner')
    joint_labels_df['index_pred'].apply(np.ceil)
    return (joint_labels_df['index'] % 1000 == joint_labels_df['index_pred'] % 1000).mean()