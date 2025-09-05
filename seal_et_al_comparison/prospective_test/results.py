import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from presentable import confusion_matrix
import numpy as np
import os
from pathlib import Path

def calculate_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculates various binary classification performance metrics.

    Args:
        y_true (pd.Series or np.array): True binary labels (0 or 1).
        y_prob (pd.Series or np.array): Predicted probabilities for the positive class (1).
        threshold (float): The probability threshold for converting probabilities to binary predictions.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Convert probabilities to binary predictions based on the threshold
    y_pred_binary = (y_prob >= threshold).astype(int)

    # Ensure y_true and y_pred_binary are aligned and have consistent types
    y_true = y_true.astype(int)
    y_pred_binary = y_pred_binary.astype(int)

    metrics = {}

    # 1. Balanced Accuracy (BA)
    metrics['BA'] = balanced_accuracy_score(y_true, y_pred_binary)

    # 2. Matthews Correlation Coefficient (MCC)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred_binary)

    # 3. Area Under Curve - Receiver Operating Characteristic (AUC-ROC)
    # Check if there's more than one class in y_true for AUC calculation
    if len(np.unique(y_true)) > 1:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['AUC-ROC'] = "N/A (Only one class in true labels)"


    # 4. Sensitivity (Recall of the positive class)
    metrics['Recall'] = recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 5. Specificity (Recall of the negative class)
    # This is equivalent to recall for pos_label=0
    metrics['Specificity'] = recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0)

    # 6. F1 Score
    metrics['F1 Score'] = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 7. Positive Predictive Value (PPV) - Precision of the positive class
    metrics['PPV'] = precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 8. Average Precision Score (AP)
    metrics['AP'] = average_precision_score(y_true, y_prob)

    # 9. Likelihood Ratio (LR+)
    # LR+ = Sensitivity / (1 - Specificity)
    specificity_complement = 1 - metrics['Specificity']
    if specificity_complement != 0:
        metrics['LR+'] = metrics['Recall'] / specificity_complement
    else:
        metrics['LR+'] = float('inf') if metrics['Recall'] > 0 else 0.0 # Handle division by zero

    return metrics

def main():
    results = {}
    df_truth = pd.read_csv("recent_dili.csv")
    df_truth = df_truth.rename(columns={"is_dili": "is_toxic"})
    df_truth['is_toxic'] = df_truth['is_toxic'].astype(int)

    df_predictions = pd.read_csv("chemeleon_predictions.csv")
    df_merged = pd.merge(df_predictions, df_truth, on='smiles', suffixes=('_pred', '_true'))
    y_true = df_merged['is_toxic_true']
    y_prob = df_merged['is_toxic_pred']
    chemeleon_threshold = 0.5
    metrics = calculate_binary_metrics(y_true, y_prob, threshold=chemeleon_threshold)
    results["chemeleon"] = metrics
    print("CheMeleon Confusion Matrix:")
    confusion_matrix(y_true, (y_prob > chemeleon_threshold).astype(int))

    df_predictions = pd.read_csv("dilipred_predictions.csv")
    df_predictions = df_predictions.rename(columns={"pred": "is_toxic"})
    df_merged = pd.merge(df_predictions, df_truth, on='smiles', suffixes=('_pred', '_true'))
    y_true = df_merged['is_toxic_true']
    y_prob = df_merged['is_toxic_pred']
    dilipred_threshold = 0.612911
    metrics = calculate_binary_metrics(y_true, y_prob, threshold=dilipred_threshold)
    results["dilipred"] = metrics
    print("dilipred Confusion Matrix:")
    confusion_matrix(y_true, (y_prob > dilipred_threshold).astype(int))

    df = pd.DataFrame(results).T
    print(df.to_markdown())
    

if __name__ == "__main__":
    main()
