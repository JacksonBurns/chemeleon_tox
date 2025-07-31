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
    metrics['Balanced Accuracy (BA)'] = balanced_accuracy_score(y_true, y_pred_binary)

    # 2. Matthews Correlation Coefficient (MCC)
    metrics['Matthews Correlation Coefficient (MCC)'] = matthews_corrcoef(y_true, y_pred_binary)

    # 3. Area Under Curve - Receiver Operating Characteristic (AUC-ROC)
    # Check if there's more than one class in y_true for AUC calculation
    if len(np.unique(y_true)) > 1:
        metrics['Area Under Curve-ROC (AUC-ROC)'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['Area Under Curve-ROC (AUC-ROC)'] = "N/A (Only one class in true labels)"


    # 4. Sensitivity (Recall of the positive class)
    metrics['Sensitivity (Recall)'] = recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 5. Specificity (Recall of the negative class)
    # This is equivalent to recall for pos_label=0
    metrics['Specificity'] = recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0)

    # 6. F1 Score
    metrics['F1 Score'] = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 7. Positive Predictive Value (PPV) - Precision of the positive class
    metrics['Positive Predictive Value (PPV)'] = precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    # 8. Average Precision Score (AP)
    metrics['Average Precision Score (AP)'] = average_precision_score(y_true, y_prob)

    # 9. Likelihood Ratio (LR+)
    # LR+ = Sensitivity / (1 - Specificity)
    specificity_complement = 1 - metrics['Specificity']
    if specificity_complement != 0:
        metrics['Likelihood Ratio (LR+)'] = metrics['Sensitivity (Recall)'] / specificity_complement
    else:
        metrics['Likelihood Ratio (LR+)'] = float('inf') if metrics['Sensitivity (Recall)'] > 0 else 0.0 # Handle division by zero

    return metrics

def main():
    results_by_strategy = {}
    df_truth = pd.read_csv("testing.csv")
    df_truth['is_toxic'] = df_truth['is_toxic'].astype(int)

    # Iterate through strategy directories
    for strategy_name in os.listdir("outputs"):
        strategy_path = os.path.join("outputs", strategy_name)
        predictions_file_path = os.path.join(strategy_path, 'test_predictions.csv')

        df_predictions = pd.read_csv(predictions_file_path)

        # Merge the dataframes on the 'smiles' column
        df_merged = pd.merge(df_predictions, df_truth, on='smiles', suffixes=('_pred', '_true'))

        # Extract true labels and predicted probabilities
        y_true = df_merged['is_toxic_true']
        y_prob = df_merged['is_toxic_pred'] # This is the probability column

        # Calculate metrics
        metrics = calculate_binary_metrics(y_true, y_prob)
        results_by_strategy[strategy_name] = metrics
    
    # also run the dilipred model results, if present
    dilipred_pred = Path("dilipred_testing_predictions.csv")
    if dilipred_pred.exists():
        df_predictions = pd.read_csv(dilipred_pred)
        df_merged = pd.merge(df_predictions, df_truth, on='smiles', suffixes=('_pred', '_true'))
        y_true = df_merged['is_toxic_true']
        y_prob = df_merged['is_toxic_pred']
        # threshold from https://github.com/Manas02/dili-pip/blob/c6aa80c0b9603cde130a48cae40d34f4fd66a1b1/dilipred/main.py#L363
        metrics = calculate_binary_metrics(y_true, y_prob, threshold=0.612911)
        results_by_strategy["dilipred_2025"] = metrics

    return results_by_strategy

if __name__ == "__main__":
    # Process all strategies and get results
    all_results = main()

    # Print all calculated metrics for each strategy
    print("\n\n--- Summary of All Strategy Results ---")
    if "Overall Error" in all_results:
        print(f"Overall Error: {all_results['Overall Error']}")
    else:
        for strategy_name, metrics in all_results.items():
            print(f"\nStrategy: {strategy_name}")
            if "Error" in metrics:
                print(f"  Error: {metrics['Error']}")
            else:
                for metric_name, value in metrics.items():
                    if isinstance(value, (float, np.floating)):
                        print(f"  {metric_name}: {value:.2f}")
                    else:
                        print(f"  {metric_name}: {value}")
