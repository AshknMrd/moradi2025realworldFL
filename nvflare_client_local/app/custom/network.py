import os
from typing import Any

import torch.nn as nn
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json


class nnUNet_CustomNetwork(nn.Module):
    """
    Wrapper class that constructs an nnU-Net v2 network instance using a local plans.json file.

    Parameters
    ----------
    num_input_channels : int
        Number of input channels for the network.
    num_output_channels : int
        Number of output channels for the network.
    allow_init : bool
        Whether network modules may perform heavy initialization.
    deep_supervision : bool
        Enable or disable deep supervision in the network.

    Notes
    -----
    This class overrides `__new__` so it returns the constructed nnU-Net network
    rather than a class instance. Functionality remains identical to the user's
    original implementation.
    """

    def __new__(
        cls,
        num_input_channels: int = 3,
        num_output_channels: int = 2,
        allow_init: bool = False,
        deep_supervision: bool = True,
    ) -> Any:
        # Resolve plans.json located in the same directory as this file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        plan_json_path = os.path.join(current_directory, "plans.json")

        plan_data = load_json(plan_json_path)
        plans_manager = PlansManager(plan_data)

        # Use the standard 3D full-resolution configuration
        configuration_manager = plans_manager.get_configuration("3d_fullres")

        # Build and return the nnU-Net network according to the plans
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=allow_init,
            deep_supervision=deep_supervision,
        )