"""flwr-ntnu-zgt: A Flower / PyTorch application"""

import json
import os
import subprocess
import warnings
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

warnings.filterwarnings("ignore", category=DeprecationWarning)


class nnUNet_CustomNetwork(nn.Module):
    """Dynamically constructs an nnU-Net network based on provided plan data."""

    def __new__(
        cls,
        num_input_channels: int = 3,
        num_output_channels: int = 2,
        allow_init: bool = False,
        deep_supervision: bool = True,
    ):
        module = import_module("flwr_ntnu_zgt")
        plan_data = module.plan_data

        plans_manager = PlansManager(plan_data)
        cfg = plans_manager.get_configuration("3d_fullres")

        return get_network_from_plans(
            cfg.network_arch_class_name,
            cfg.network_arch_init_kwargs,
            cfg.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=allow_init,
            deep_supervision=deep_supervision,
        )


def train(client_name, epochs, folds, checkpoint, plan_identifier, configuration, client_base_dir):
    """Run preprocessing and training for a given client."""

    os.environ["nnUNet_raw"] = os.path.join(client_base_dir, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(client_base_dir, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(client_base_dir, "nnUNet_results")

    nnUNet_raw = os.getenv("nnUNet_raw")
    nnUNet_preprocessed = os.getenv("nnUNet_preprocessed")
    nnUNet_results = os.getenv("nnUNet_results")

    # Determine dataset/task identifiers
    if "ntnu" in client_name:
        task_name, task_id = "Dataset103_ntnu_client", "103"
    elif "zgt" in client_name:
        task_name, task_id = "Dataset102_zgt_client", "102"
    elif "cent" in client_name:
        task_name, task_id = "Dataset105_centralized", "105"
    else:
        task_name, task_id = None, None

    trainer = f"nnUNetTrainerCELoss_{epochs}epochs"

    dataset_dir = os.path.join(nnUNet_raw, task_name)
    dataset_json = os.path.join(dataset_dir, "dataset.json")
    create_dataset_json(dataset_dir, dataset_json)

    fl_chkpoint_dir = os.path.join(client_base_dir, f"checkpoint_fl_{client_name}.pth")

    labels_dir = os.path.join(nnUNet_raw, task_name, "labelsTr")
    nii_gz_files = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]

    print(f"\nClient: {client_name}")
    print(f"{task_name} has {len(nii_gz_files)} cases.")
    print(f"nnUNet_raw: {nnUNet_raw}")
    print(f"nnUNet_preprocessed: {nnUNet_preprocessed}")
    print(f"nnUNet_results: {nnUNet_results}")
    print(f"Available device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    checkpoint_dir = os.path.join(
        nnUNet_results,
        task_name,
        f"{trainer}__{plan_identifier}__{configuration}",
        f"fold_{folds}",
        checkpoint,
    )

    # Training logic
    if os.path.exists(os.path.join(nnUNet_preprocessed, task_name)):
        model_dir = os.path.join(
            nnUNet_results, task_name, f"{trainer}__{plan_identifier}__{configuration}"
        )

        if os.path.exists(model_dir):
            if os.path.exists(fl_chkpoint_dir):
                print("Results & preprocessing found. Loading FL checkpoint...\n")
                run_command(
                    f"nnUNetv2_train {task_id} {configuration} {folds} "
                    f"-tr {trainer} -pretrained_weights {fl_chkpoint_dir} --npz"
                )
            else:
                print("The model is already trained. No FL checkpoint found.\n")
        else:
            print("Preprocessing found. Starting training...\n")
            run_command(
                f"nnUNetv2_train {task_id} {configuration} {folds} "
                f"-tr {trainer} --npz"
            )
    else:
        print("Starting preprocessing...\n")
        run_command(
            f"nnUNetv2_plan_and_preprocess -d {task_id} --dataset_json {dataset_json} "
            f"--verify_dataset_integrity -c {configuration}"
        )
        print("Starting training...\n")
        run_command(
            f"nnUNetv2_train {task_id} {configuration} {folds} "
            f"-tr {trainer} --npz"
        )

    checkpoint_data = torch.load(checkpoint_dir, weights_only=False)
    train_loss = float(np.sum(checkpoint_data["logging"]["train_losses"]))
    return round(train_loss, 5)


def test(client_name, epochs, folds, checkpoint, plan_identifier, configuration, client_base_dir):
    """Evaluate validation loss and EMA dice score."""

    _, checkpoint_dir = get_chkpoint_dir_from_config(
        client_name, epochs, folds, checkpoint, plan_identifier, configuration, client_base_dir
    )

    checkpoint_data = torch.load(checkpoint_dir, weights_only=False)
    val_loss = np.sum(checkpoint_data["logging"]["val_losses"])
    ema_dice = checkpoint_data["logging"]["ema_fg_dice"]
    ema_dice = 0 if ema_dice is None else np.sum(ema_dice)

    return round(float(val_loss), 5), round(float(ema_dice), 5)


def set_weights(fl_parameters, fl_chkpoint_dir, checkpoint_dir):
    """Replace local model weights with federated weights."""

    checkpoint_data = torch.load(checkpoint_dir, weights_only=False)
    model_params = checkpoint_data["network_weights"]

    for idx, key in enumerate(model_params.keys()):
        model_params[key] = torch.tensor(fl_parameters[idx])

    checkpoint_data["network_weights"] = model_params

    torch.save(checkpoint_data, checkpoint_dir)
    torch.save(checkpoint_data, fl_chkpoint_dir)


def get_weights(checkpoint_dir):
    """Extract model weights as numpy arrays."""

    checkpoint_data = torch.load(checkpoint_dir, weights_only=False)
    network_parameters = checkpoint_data["network_weights"]
    return [v.cpu().numpy() for v in network_parameters.values()]


def run_command(command: str):
    """Execute a shell command and print stderr on failure."""
    try:
        subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")


def create_dataset_json(dataset_dir, output_dir):
    """Create a dataset.json file based on available training labels."""

    label_dir = os.path.join(dataset_dir, "labelsTr")
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".nii.gz")])

    client_name = os.path.basename(dataset_dir)
    data_center = client_name.split("_")[-1]

    dataset_dict = {
        "channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(label_files),
        "file_ending": ".nii.gz",
        "name": f"picai_{data_center}_nnunetv2",
        "reference": "none",
        "release": "1.0",
        "description": "bpMRI scans from PI-CAI dataset to train by nnUNetv2",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with open(output_dir, "w") as f:
        json.dump(dataset_dict, f, indent=4)


def get_chkpoint_dir_from_config(
    client_name,
    epochs,
    folds,
    checkpoint,
    plan_identifier,
    configuration,
    client_base_dir,
    len_data=False,
):
    """Return checkpoint locations, and optionally dataset length."""

    task_name = "Dataset103_ntnu_client" if "ntnu" in client_name else \
                "Dataset102_zgt_client" if "zgt" in client_name else None

    trainer = f"nnUNetTrainerCELoss_{epochs}epochs"
    nnUNet_results = os.path.join(client_base_dir, "nnUNet_results")
    fl_chkpoint_dir = os.path.join(client_base_dir, f"checkpoint_fl_{client_name}.pth")

    checkpoint_dir = os.path.join(
        nnUNet_results,
        task_name,
        f"{trainer}__{plan_identifier}__{configuration}",
        f"fold_{folds}",
        checkpoint,
    )

    if len_data:
        nnUNet_raw = os.path.join(client_base_dir, "nnUNet_raw")
        labels_dir = os.path.join(nnUNet_raw, task_name, "labelsTr")
        count = len([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])
        return fl_chkpoint_dir, checkpoint_dir, count

    return fl_chkpoint_dir, checkpoint_dir