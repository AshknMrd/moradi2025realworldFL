"""flwr-ntnu-zgt: A Flower/PyTorch server application."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flwr_ntnu_zgt.strategy import CustomFedAvg
from flwr_ntnu_zgt.task import nnUNet_CustomNetwork

# Suppress specific deprecated warnings (optional, minimal impact on functionality)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the Flower server components.

    Parameters
    ----------
    context : Context
        Flower runtime context containing configuration and environment details.

    Returns
    -------
    ServerAppComponents
        Configured strategy and server settings for the Flower server.
    """
    run_cfg = context.run_config

    # Load configuration values
    num_rounds = run_cfg["num-server-rounds"]
    fraction_fit = run_cfg["fraction-fit"]
    min_available_clients = run_cfg["min-available-clients"]
    fraction_evaluate = run_cfg["fraction-evaluate"]
    min_fit_clients = run_cfg["min-fit-clients"]
    min_evaluate_clients = run_cfg["min-evaluate-clients"]
    client_base_dir = run_cfg["client-base-dir"]

    # Initialize model parameters
    model = nnUNet_CustomNetwork()
    ndarrays = [tensor.cpu().numpy() for _, tensor in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Configure strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        initial_parameters=parameters,
        client_base_dir=client_base_dir,
    )

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)