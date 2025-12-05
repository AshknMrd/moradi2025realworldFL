"""
flwr-ntnu-zgt: A Flower / PyTorch application.
"""

import json
import warnings
from pathlib import Path

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from flwr_ntnu_zgt.task import (
    nnUNet_CustomNetwork,
    get_chkpoint_dir_from_config,
    get_weights,
    set_weights,
    test,
    train,
)

# Suppress deprecation warnings to avoid unnecessary console clutter
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FlowerClient(NumPyClient):
    """Federated learning client for Flower using a custom nnU-Net model."""

    def __init__(
        self,
        net: torch.nn.Module,
        client_name: str,
        epochs: int,
        folds: int,
        checkpoint: str,
        plan_identifier: str,
        configuration: dict,
        client_base_dir: str,
    ) -> None:
        self.net = net
        self.client_name = client_name
        self.epochs = epochs
        self.folds = folds
        self.checkpoint = checkpoint
        self.plan_identifier = plan_identifier
        self.configuration = configuration
        self.client_base_dir = client_base_dir

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Train the model on the client dataset."""
        fl_chkpt_dir, chkpt_dir, data_len = get_chkpoint_dir_from_config(
            self.client_name,
            self.epochs,
            self.folds,
            self.checkpoint,
            self.plan_identifier,
            self.configuration,
            self.client_base_dir,
            len_data=True,
        )

        if Path(chkpt_dir).exists():
            set_weights(parameters, fl_chkpt_dir, chkpt_dir)

        train_loss = train(
            self.client_name,
            self.epochs,
            self.folds,
            self.checkpoint,
            self.plan_identifier,
            self.configuration,
            self.client_base_dir,
        )

        return (
            get_weights(chkpt_dir),
            data_len,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the client test dataset."""
        fl_chkpt_dir, chkpt_dir, data_len = get_chkpoint_dir_from_config(
            self.client_name,
            self.epochs,
            self.folds,
            self.checkpoint,
            self.plan_identifier,
            self.configuration,
            self.client_base_dir,
            len_data=True,
        )

        set_weights(parameters, fl_chkpt_dir, chkpt_dir)

        loss, dice = test(
            self.client_name,
            self.epochs,
            self.folds,
            self.checkpoint,
            self.plan_identifier,
            self.configuration,
            self.client_base_dir,
        )

        # The +1 preserves original behaviour
        return loss, (int(data_len * 0.2) + 1), {"dice": dice}


def client_fn(context: Context):
    """Create and return the Flower client instance."""

    partition_id = context.node_config["partition-id"]
    client_names = json.loads(context.run_config["client-names"])
    client_name = client_names[str(partition_id)]

    net = nnUNet_CustomNetwork()

    return FlowerClient(
        net=net,
        client_name=client_name,
        epochs=context.run_config["epochs"],
        folds=context.run_config["folds"],
        checkpoint=context.run_config["checkpoint"],
        plan_identifier=context.run_config["plan-identifier"],
        configuration=context.run_config["configuration"],
        client_base_dir=context.run_config["client-base-dir"],
    ).to_client()


# Flower Client Application
app = ClientApp(client_fn)