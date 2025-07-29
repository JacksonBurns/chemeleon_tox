import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import numpy as np
import argparse
import os  # Import os module to check file existence


def calculate_multiclass_metrics(predictions_filepath, true_labels_filepath):
    """
    Calculates multiclass classification performance metrics from two CSV files.

    Args:
        predictions_filepath (str): The file path to the CSV containing model predictions,
                                    with columns 'smiles', 'dilirank' (predicted class),
                                    and 'dilirank_prob' (class probabilities).
        true_labels_filepath (str): The file path to the CSV containing true class labels,
                                    with columns 'smiles' and 'dilirank' (true class).

    Returns:
        dict: A dictionary containing the calculated metrics, or an error message.
    """
    # Check if files exist before attempting to read
    if not os.path.exists(predictions_filepath):
        return {"Error": f"Predictions file not found: {predictions_filepath}"}
    if not os.path.exists(true_labels_filepath):
        return {"Error": f"True labels file not found: {true_labels_filepath}"}

    # Read the CSV data from the provided file paths
    df_predictions = pd.read_csv(predictions_filepath)
    df_true_labels = pd.read_csv(true_labels_filepath)

    is_binary = "binary" in predictions_filepath
    if is_binary:
        target = "diliclass"
        df_predictions[target] = (df_predictions[target] > 0.5).astype(int)
    else:
        target = "dilirank"

    # Merge the dataframes on the 'smiles' column
    # This ensures that predictions and true labels are aligned for each molecule
    df_merged = pd.merge(
        df_predictions, df_true_labels, on="smiles", suffixes=("_pred", "_true")
    )

    # Check if the merged DataFrame is empty after merging
    if df_merged.empty:
        return {
            "Error": "No common 'smiles' identifiers found between the two files. "
            "Please ensure 'smiles' columns match for merging."
        }

    # Extract true labels and predicted labels
    y_true = df_merged[f"{target}_true"]
    y_pred = df_merged[f"{target}_pred"]

    if not is_binary:
        # Parse the 'dilirank_prob' string into a numpy array of floats
        # Each string is like "0.20822667,0.49869585,0.29307747"
        # We need to convert it to a list of floats for each row
        y_prob_str = df_merged[f"{target}_prob"].apply(
            lambda x: [float(p) for p in x.split(",")]
        )
        # Convert list of lists to a 2D numpy array
        y_prob = np.array(y_prob_str.tolist())

    # --- Calculate Classification Metrics ---

    # Accuracy: Proportion of correctly classified instances
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1-score:
    # For multiclass, 'weighted' averages the metric for each class,
    # weighted by the number of true instances for each class.
    # 'macro' calculates metrics for each label, and finds their unweighted mean.
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion Matrix: A table used to describe the performance of a classification model
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Classification Report: Builds a text report showing the main classification metrics
    # per class.
    class_report = classification_report(y_true, y_pred, zero_division=0)

    # ROC AUC Score:
    # For multiclass, 'ovr' (One-vs-Rest) calculates the AUC for each class against all others.
    # Requires probability estimates.
    # Determine the number of unique classes from y_true
    unique_classes = np.sort(y_true.unique())
    num_classes = len(unique_classes)

    results = {
        "Accuracy": accuracy,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1-Score (Weighted)": f1_weighted,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-Score (Macro)": f1_macro,
        "Confusion Matrix": conf_matrix.tolist(),  # Convert numpy array to list for easier printing
        "Classification Report": class_report,
    }

    # Check if the number of columns in y_prob matches the number of unique classes
    if not is_binary:
        if y_prob.shape[1] != num_classes:
            print(
                f"Warning: Number of probability columns ({y_prob.shape[1]}) does not match "
                f"number of unique true classes ({num_classes}). ROC AUC might be inaccurate "
                f"or fail. Ensure probabilities are ordered by class labels (e.g., 0, 1, 2...)."
            )
            roc_auc_ovr_weighted = "N/A (Mismatch in prob columns and classes)"
            roc_auc_ovr_macro = "N/A (Mismatch in prob columns and classes)"
        else:
            try:
                roc_auc_ovr_weighted = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
                roc_auc_ovr_macro = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            except ValueError as e:
                roc_auc_ovr_weighted = f"Error: {e}"
                roc_auc_ovr_macro = f"Error: {e}"

        results["ROC AUC (OVR, Weighted)"] = roc_auc_ovr_weighted
        results["ROC AUC (OVR, Macro)"] = roc_auc_ovr_macro

    return results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate multiclass classification performance metrics."
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the CSV file containing model predictions (with 'dilirank' and 'dilirank_prob').",
    )
    parser.add_argument(
        "--true-labels-file",
        type=str,
        required=True,
        help="Path to the CSV file containing true class labels (with 'dilirank').",
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.exists(args.predictions_file):
        raise RuntimeError(f"File {args.predictions_file} not found.")

    if not os.path.exists(args.true_labels_file):
        raise RuntimeError(f"File {args.true_labels} not found.")
    # Calculate the metrics using the provided file paths
    metrics = calculate_multiclass_metrics(args.predictions_file, args.true_labels_file)

    # Print the results
    if "Error" in metrics:
        print(metrics["Error"])
    else:
        print("\n--- Classification Performance Metrics ---")
        for key, value in metrics.items():
            if key == "Confusion Matrix":
                print(
                    f"\n{key}:\n{np.array(value)}"
                )  # Print as numpy array for better readability
            elif key == "Classification Report":
                print(f"\n{key}:\n{value}")
            elif isinstance(
                value, (float, np.floating)
            ):  # Check if it's a float or numpy float
                print(f"{key}: {value:.4f}")  # Format floats to 4 decimal places
            else:
                print(f"{key}: {value}")  # Print other types directly
