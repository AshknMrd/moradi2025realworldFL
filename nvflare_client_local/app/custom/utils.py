import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc


def load_json(config_path: str) -> dict:
    """
    Load and return a JSON file as a dictionary.

    """
    with open(config_path, "r") as f:
        return json.load(f)


def return_network_weights_val_loss(checkpoint_dir: str):
    """
    Load a model checkpoint and return the stored network weights and validation loss.

    """
    checkpoint_data = torch.load(checkpoint_dir, weights_only=False)
    val_loss = -np.mean(checkpoint_data["logging"]["val_losses"])
    network_parameters = checkpoint_data["network_weights"]
    return network_parameters, val_loss


def return_validation_metrics(summary_json_dir: str):
    """
    Compute AUROC, Average Precision (AP), PICAI score, and mean Dice from validation metrics.

    """
    summary_data = load_json(summary_json_dir)

    tpr_list, fpr_list = [], []
    precision_list, recall_list = [], []

    for case_data in summary_data["metric_per_case"]:
        metrics = case_data["metrics"]["1"]
        tp, fn, fp, tn = metrics["TP"], metrics["FN"], metrics["FP"], metrics["TN"]

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr  # recall = TPR

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)

    # Sort values for correct AUC computation
    sorted_fpr, sorted_tpr = zip(*sorted(zip(fpr_list, tpr_list)))
    sorted_recall, sorted_precision = zip(*sorted(zip(recall_list, precision_list)))

    auroc = auc(sorted_fpr, sorted_tpr)
    ap = auc(sorted_recall, sorted_precision)
    picai_score = (auroc + ap) / 2
    mean_dice = summary_data["mean"]["1"]["Dice"]

    return auroc, ap, picai_score, mean_dice


def run_command(command: str) -> None:
    """
    Execute a shell command and print errors if the command fails.

    Parameters
    ----------
    command : str
        Shell command to execute.
    """
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")