import re
import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import pandas as pd


def parse_metrics(file_path):
    print(f"Parsing metrics from {file_path}")
    metrics = defaultdict(dict)
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse hallucinations/errors
    metrics['hallucinations_errors'] = int(re.search(r'hallucinations/errors:\s(\d+)', lines[0]).group(1))
    metrics['responses_with_hallucinations'] = int(re.search(r'responses with some hallucinated portion:\s(\d+)', lines[1]).group(1))
    
    # Parse Encodings
    encodings_start = lines.index("Encodings:\n")
    metrics['true_encodings'] = {
        'class_0': int(re.search(r'Class 0:\s(\d+)', lines[encodings_start + 2]).group(1)),
        'class_1': int(re.search(r'Class 1:\s(\d+)', lines[encodings_start + 3]).group(1))
    }
    metrics['predicted_encodings'] = {
        'class_0': int(re.search(r'Class 0:\s(\d+)', lines[encodings_start + 5]).group(1)),
        'class_1': int(re.search(r'Class 1:\s(\d+)', lines[encodings_start + 6]).group(1))
    }
    
    # Parse Performance Metrics
    performance_start = lines.index("Performance Metrics:\n")
    metrics['accuracy'] = float(re.search(r'Accuracy:\s([\d.]+)', lines[performance_start + 1]).group(1))
    metrics['macro_averages'] = {
        'f1_score': float(re.search(r'F1 Score:\s([\d.]+)', lines[performance_start + 3]).group(1)),
        'precision': float(re.search(r'Precision:\s([\d.]+)', lines[performance_start + 4]).group(1)),
        'recall': float(re.search(r'Recall:\s([\d.]+)', lines[performance_start + 5]).group(1))
    }
    metrics['weighted_averages'] = {
        'f1_score': float(re.search(r'F1 Score:\s([\d.]+)', lines[performance_start + 7]).group(1)),
        'precision': float(re.search(r'Precision:\s([\d.]+)', lines[performance_start + 8]).group(1)),
        'recall': float(re.search(r'Recall:\s([\d.]+)', lines[performance_start + 9]).group(1))
    }
    
    # Parse Confusion Matrix
    try:
        confusion_matrix_start = lines.index("Confusion Matrix:\n") + 1
        tp, tp_pct = map(float, re.search(r'TP:\s(\d+)\s\(([\d.]+)%\)', lines[confusion_matrix_start]).groups())
        fp, fp_pct = map(float, re.search(r'FP:\s(\d+)\s\(([\d.]+)%\)', lines[confusion_matrix_start]).groups())
        fn, fn_pct = map(float, re.search(r'FN:\s(\d+)\s\(([\d.]+)%\)', lines[confusion_matrix_start + 1]).groups())
        tn, tn_pct = map(float, re.search(r'TN:\s(\d+)\s\(([\d.]+)%\)', lines[confusion_matrix_start + 1]).groups())
        metrics['confusion_matrix'] = {
            'TP': {'count': tp, 'percentage': tp_pct},
            'FP': {'count': fp, 'percentage': fp_pct},
            'FN': {'count': fn, 'percentage': fn_pct},
            'TN': {'count': tn, 'percentage': tn_pct}
        }
    except:
        pass
    
    return metrics


def parse_all_metrics(base_dir):
    results = {}
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "metrics_and_model.txt":
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_dir)
                metrics = parse_metrics(file_path)
                results[relative_path] = metrics
    
    return results

def calculate_auc(metrics):
    """
    Calculate AUC for binary classification based on metrics.

    Args:
        metrics (dict): A dictionary containing confusion matrix values.
    
    Returns:
        float: The calculated AUC value.
    """
    # Extract confusion matrix values
    tp = metrics['confusion_matrix']['TP']['count']
    fp = metrics['confusion_matrix']['FP']['count']
    fn = metrics['confusion_matrix']['FN']['count']
    tn = metrics['confusion_matrix']['TN']['count']

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity or Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Generate a simulated binary classification output for AUC calculation
    y_true = [1] * int(tp + fn) + [0] * int(fp + tn)  # True labels
    y_scores = [1] * int(tp) + [0] * int(fn) + [1] * int(fp) + [0] * int(tn)  # Predicted scores
    if len(set(y_true)) == 1:
        # Only one class present in y_true, AUC is undefined
        return None

    # Calculate AUC using sklearn's roc_auc_score
    auc = roc_auc_score(y_true, y_scores)
    
    return auc

def calculate_all_auc(all_metrics):
    """
    Calculate AUC for all metrics in a dataset.

    Args:
        all_metrics (dict): A dictionary containing metrics for multiple files.
    
    Returns:
        dict: A dictionary with file paths as keys and AUC values as values.
    """
    auc_results = {}
    for file_path, metrics in all_metrics.items():
        auc = calculate_auc(metrics)
        auc_results[file_path] = auc
    
    return auc_results

def prepare_data_for_excel(all_metrics, auc_results):
    data = []
    for file_path, metrics in all_metrics.items():
        confusion_matrix = metrics['confusion_matrix']
        tp = confusion_matrix['TP']['count']
        fp = confusion_matrix['FP']['count']
        fn = confusion_matrix['FN']['count']
        tn = confusion_matrix['TN']['count']

        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Get other metrics
        errors = metrics['hallucinations_errors']
        responses_with_hallucinations = metrics['responses_with_hallucinations']
        hallucination_rate = responses_with_hallucinations/(tp + fp) if (tp + fp) > 0 else 1
        accuracy = metrics.get('accuracy', None)
        macro_f1 = metrics['macro_averages']['f1_score']

        weighted_f1 = metrics['weighted_averages']['f1_score']
        weighted_precision = metrics['weighted_averages']['precision']
        weighted_recall = metrics['weighted_averages']['recall']
        auc = auc_results.get(file_path, None)

        # Add row to data
        data.append({
            'File Path': file_path,
            'Errors': errors,
            'Responses with Hallucinations': responses_with_hallucinations,
            'Hallucination Rate': hallucination_rate,
            'Accuracy': accuracy,
            'Macro F1': macro_f1,
            'Weighted F1': weighted_f1,
            'Weighted Precision': weighted_precision,
            'Weighted Recall': weighted_recall,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'TPR': tpr,
            'FPR': fpr,
            'AUC': auc
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def save_to_excel(dataframe, output_file):
    dataframe.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

def collect_codes_metrics(base_dir):
    results = {}
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if "_codes" in file:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_dir)
                metrics = parse_metrics(file_path)  # Ensure parse_metrics can handle _codes files
                results[relative_path] = metrics
    
    return results


if __name__ == "__main__":
    all_metrics = parse_all_metrics('val_set_results')
    all_auc = calculate_all_auc(all_metrics)

    # Prepare and save data
    df = prepare_data_for_excel(all_metrics, all_auc)
    output_file = 'val_set_results_metrics.xlsx'
    save_to_excel(df, output_file)