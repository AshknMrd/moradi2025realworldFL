import os
from typing import List, Optional, Union

import torch.cuda
from network import nnUNet_CustomNetwork
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTConstants:
    """Constants used for PyTorch model location."""

    PTServerName = "server"
    PTFileModelName = "model_parameters.pt"


class PTModelLocator(ModelLocator):
    """Model locator for loading a PyTorch nnUNet model in NVFLARE."""

    def __init__(
        self,
        exclude_vars: Optional[List[str]] = None,
        model=None,
        use_external_file: bool = False,
    ) -> None:
        super().__init__()

        # Note: `model` argument is currently not used to preserve existing behavior.
        self.model = nnUNet_CustomNetwork()
        self.exclude_vars = exclude_vars
        self.use_external_file = use_external_file

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Return the list of known model names."""
        return [PTConstants.PTServerName]

    def locate_model(
        self,
        model_name: str,
        fl_ctx: FLContext,
    ) -> Union[DXO, None]:
        """Locate and load the model corresponding to the given name.

        Returns:
            DXO for the model if found and successfully loaded, otherwise None.
        """
        if model_name != PTConstants.PTServerName:
            self.log_error(
                fl_ctx,
                f"PTModelLocator doesn't recognize name: {model_name}",
                fire_event=False,
            )
            return None

        try:
            if self.use_external_file:
                # Use an externally mounted model file.
                model_path = os.path.join(
                    "/external_files/mimic-cxr",
                    os.path.basename(PTConstants.PTFileModelName),
                )
            else:
                # Use the model file from the server's workspace.
                server_run_dir = (
                    fl_ctx.get_engine()
                    .get_workspace()
                    .get_app_dir(fl_ctx.get_run_number())
                )
                model_path = os.path.join(server_run_dir, PTConstants.PTFileModelName)

            if not os.path.exists(model_path):
                return None

            # Load the torch model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data = torch.load(model_path, map_location=device)

            # Setup the persistence manager configuration.
            if self.model:
                default_train_conf = {"train": {"model": type(self.model).__name__}}
            else:
                default_train_conf = None

            # Use persistence manager to get a model_learnable
            persistence_manager = PTModelPersistenceFormatManager(
                data,
                default_train_conf=default_train_conf,
            )
            ml = persistence_manager.to_model_learnable(exclude_vars=None)

            # Convert to DXO and return
            return model_learnable_to_dxo(ml)

        except Exception as e:
            self.log_error(
                fl_ctx,
                f"Error in retrieving {model_name}: {e}",
                fire_event=False,
            )
            return None