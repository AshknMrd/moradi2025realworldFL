import json
import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates
from utils import *


def assign_new_model_parameters(
    fl_parameters: Dict[str, Any],
    fl_checkpoint_path: str,
    checkpoint_path: str,
) -> None:
    """
    Replace the parameters inside a checkpoint file with FL-provided parameters.

    Args:
        fl_parameters: Dict of global model parameters from FL server.
        fl_checkpoint_path: Path to save updated FL checkpoint.
        checkpoint_path: Path of the existing client model checkpoint.
    """
    client_checkpoint_data = torch.load(checkpoint_path, weights_only=False)
    client_model_parameters = client_checkpoint_data["network_weights"]

    for key in client_model_parameters:
        client_model_parameters[key] = torch.tensor(fl_parameters[key])

    client_checkpoint_data["network_weights"] = client_model_parameters

    # Save locally and in FL checkpoint directory
    torch.save(client_checkpoint_data, checkpoint_path)
    torch.save(client_checkpoint_data, fl_checkpoint_path)


class NNUNetExecutor(Executor):
    """
    NVFlare Executor for orchestrating nnU-Net v2 federated training.
    """

    def __init__(
        self,
        epochs: int,
        folds: int,
        checkpoint: str,
        plan_identifier: str,
        configuration: str,
        client_input_base_dir: str,
        client_output_base_dir: str,
        evaluation_mode: str,
        **kwargs,
    ):
        super().__init__()

        self.epochs = epochs
        self.folds = folds
        self.checkpoint = checkpoint
        self.plan_identifier = plan_identifier
        self.configuration = configuration
        self.client_input_base_dir = client_input_base_dir
        self.client_output_base_dir = client_output_base_dir
        self.evaluation_mode = evaluation_mode

        self.trainer = f"nnUNetTrainerCELoss_{self.epochs}epochs"

        # Set nnU-Net expected environment variables
        os.environ["nnUNet_raw"] = os.path.join(client_input_base_dir, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(
            client_output_base_dir, "nnUNet_preprocessed"
        )
        os.environ["nnUNet_results"] = os.path.join(
            client_output_base_dir, "nnUNet_results"
        )

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_context: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Main execution method called by NVFlare per FL round.
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        client_id = fl_context.get_identity_name()

        # Dataset identification based on client ID
        if "ntnu" in client_id:
            self.task_name_cfg, self.task_id = "Dataset103_ntnu_client", "103"
        elif "zgt" in client_id:
            self.task_name_cfg, self.task_id = "Dataset102_zgt_client", "102"
        elif "cent" in client_id:
            self.task_name_cfg, self.task_id = "Dataset105_centralized", "105"
        else:
            self.task_name_cfg, self.task_id = None, "104"

        nnunet_raw = os.getenv("nnUNet_raw")
        nnunet_preprocessed = os.getenv("nnUNet_preprocessed")
        nnunet_results = os.getenv("nnUNet_results")

        dataset_dir = os.path.join(nnunet_raw, self.task_name_cfg)
        dataset_json = os.path.join(dataset_dir, "dataset.json")

        current_round_num = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_round_num = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        self.current_round_num = current_round_num
        self.total_round_num = total_round_num

        os.makedirs(
            os.path.join(self.client_input_base_dir, "model_parameters"), exist_ok=True
        )
        self.fl_chkpoint_dir = os.path.join(
            self.client_output_base_dir,
            "model_parameters",
            f"checkpoint_fl_{client_id}.pth",
        )

        # Count how many training cases exist
        label_dir = os.path.join(nnunet_raw, self.task_name_cfg, "labelsTr")
        self.num_cases = len([f for f in os.listdir(label_dir) if f.endswith(".nii.gz")])

        print(f"\nClient {client_id} at round {current_round_num+1}/{total_round_num}:\n")

        if current_round_num == 0:
            print(f"Client {client_id} has {self.num_cases} cases.")
            print(f"nnUNet_raw: {nnunet_raw}")
            print(f"nnUNet_preprocessed: {nnunet_preprocessed}")
            print(f"nnUNet_results: {nnunet_results}\n")

            avail_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Available device: {avail_device}")

        checkpoint_path = os.path.join(
            nnunet_results,
            self.task_name_cfg,
            f"{self.trainer}__{self.plan_identifier}__{self.configuration}",
            f"fold_{self.folds}",
            self.checkpoint,
        )

        summary_json_path = os.path.join(
            nnunet_results,
            self.task_name_cfg,
            f"{self.trainer}__{self.plan_identifier}__{self.configuration}",
            f"fold_{self.folds}",
            "validation",
            "summary.json",
        )

        # Receive global model weights
        if shareable:
            dxo = from_shareable(shareable)
            if dxo.data_kind == DataKind.WEIGHTS and os.path.exists(checkpoint_path):
                assign_new_model_parameters(
                    dxo.data, self.fl_chkpoint_dir, checkpoint_path
                )

        # Local training
        try:
            self.run_training(
                self.task_id,
                self.configuration,
                self.folds,
                self.trainer,
                self.plan_identifier,
                self.fl_chkpoint_dir,
                nnunet_raw,
                nnunet_preprocessed,
                nnunet_results,
                self.task_name_cfg,
                current_round_num,
                total_round_num,
                dataset_json,
            )
        except Exception as e:
            self.log_exception(fl_context, f"Exception during training: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Evaluation section
        try:
            trained_parameters, val_loss = return_network_weights_val_loss(
                checkpoint_path
            )
            auroc, ap, picai_score, mean_dice = return_validation_metrics(
                summary_json_path
            )
        except Exception as e:
            self.log_exception(fl_context, f"Exception during evaluation: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Build shareable DXO
        trained_parameters = {
            key: value.detach().cpu().numpy()
            for key, value in trained_parameters.items()
        }

        metrics = {
            "val_loss": val_loss,
            "mean_dice": mean_dice,
            "auroc": auroc,
            "ap": ap,
            "picai_score": picai_score,
            "NUM_STEPS_CURRENT_ROUND": self.epochs,
            "Aggregation_weight": self.num_cases,
        }

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trained_parameters, meta=metrics)
        return dxo.to_shareable()

    def run_training(
        self,
        task_id: str,
        configuration: str,
        folds: int,
        trainer: str,
        plan_identifier: str,
        fl_chkpoint_dir: str,
        nnunet_raw: str,
        nnunet_preprocessed: str,
        nnunet_results: str,
        task_name_cfg: str,
        current_round: int,
        total_round: int,
        dataset_json: str,
    ) -> None:
        """
        Handle nnU-Net preprocessing, training, and evaluation dependent on state.
        """
        preprocessed_exists = os.path.exists(
            os.path.join(nnunet_preprocessed, self.task_name_cfg)
        )

        model_dir = os.path.join(
            nnunet_results,
            self.task_name_cfg,
            f"{self.trainer}__{self.plan_identifier}__{self.configuration}",
        )

        if preprocessed_exists:
            # Pre-processing exists
            if os.path.exists(model_dir):
                if os.path.exists(self.fl_chkpoint_dir):
                    print("Results & Pre-processing exist, loading FL checkpoint...\n")
                    run_command(
                        f"nnUNetv2_train {self.task_id} {self.configuration} {self.folds} "
                        f"-tr {self.trainer} -pretrained_weights {self.fl_chkpoint_dir} "
                        f"-current_round {self.current_round_num} -total_round {self.total_round_num} --npz"
                    )
                else:
                    print("Model exists and no FL checkpoint found.\n")
            else:
                print("Pre-processing exists, starting training...\n")
                run_command(
                    f"nnUNetv2_train {self.task_id} {self.configuration} {self.folds} "
                    f"-tr {self.trainer} -current_round {self.current_round_num} "
                    f"-total_round {self.total_round_num} --npz"
                )

        else:
            # No pre-processing
            print("Starting Pre-processing...\n")
            run_command(
                f"nnUNetv2_plan_and_preprocess -d {self.task_id} "
                f"--dataset_json {dataset_json} --verify_dataset_integrity -c {self.configuration}"
            )
            print("Starting Training...\n")
            run_command(
                f"nnUNetv2_train {self.task_id} {self.configuration} {self.folds} "
                f"-tr {self.trainer} -current_round {self.current_round_num} "
                f"-total_round {self.total_round_num} --npz"
            )

        # Evaluation after final round
        if current_round + 1 == total_round:
            if self.evaluation_mode == "nnunet_eval":
                summary_json_path = os.path.join(
                    nnunet_results,
                    self.task_name_cfg,
                    f"{self.trainer}__{self.plan_identifier}__{self.configuration}",
                    f"fold_{self.folds}",
                    "validation",
                    "summary.json",
                )
                auroc, ap, picai_score, _ = return_validation_metrics(
                    summary_json_path
                )

            elif self.evaluation_mode == "picai_eval_lesion_extract":
                val_gt_folder = os.path.join(nnunet_raw, self.task_name_cfg, "labelsTr")
                val_predictions = os.path.join(
                    nnunet_results,
                    self.task_name_cfg,
                    f"{self.trainer}__{self.plan_identifier}__{self.configuration}",
                    f"fold_{self.folds}",
                    "validation",
                )
                metrics = evaluate_folder(
                    y_det_dir=val_predictions,
                    y_true_dir=val_gt_folder,
                    pred_extensions=[".npz"],
                    y_det_postprocess_func=lambda pred: extract_lesion_candidates(
                        pred, threshold="dynamic"
                    )[0],
                )
                auroc, ap, picai_score = metrics.auroc, metrics.AP, metrics.score

            print("\n")
            print(f"Evaluation mode: {self.evaluation_mode}")
            print(f"AUROC: {round(auroc, 3)}")
            print(f"AP: {round(ap, 3)}")
            print(f"PICAI Score: {round(picai_score, 3)}\n")