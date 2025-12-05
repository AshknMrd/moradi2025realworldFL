import datetime
import os
from pathlib import Path
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class PTFileModelPersistorAllCheckpoints(PTFileModelPersistor):
    """A PTFileModelPersistor that additionally saves every checkpoint with a timestamped filename."""

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        """Save the model using the base persistor and also archive a timestamped checkpoint."""
        super().save_model(ml, fl_ctx)

        dest_dir = Path("/workdir_nvflare/model_parameters")
        dest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        checkpoint_path = dest_dir / f"checkpoint_{timestamp}.pt"

        # Save model parameters to the new checkpoint file
        self.save_model_file(str(checkpoint_path))

        self.logger.info(f"Saved model parameters to {checkpoint_path}")