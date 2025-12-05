from pathlib import Path
from datetime import datetime
import json
import torch

from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from flwr_ntnu_zgt.task import nnUNet_CustomNetwork


def create_run_dir(base_dir: str) -> Path:
    """Create and return a timestamped output directory for the current run."""
    run_dir = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    save_path = Path(base_dir) / "outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path

class CustomFedAvg(FedAvg):
    """Custom Flower FedAvg strategy with model saving and result logging."""

    def __init__(self, client_base_dir: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_base_dir = client_base_dir
        self.save_path = create_run_dir(client_base_dir)
        self.results = {}

    def save_fl_model(self, server_round: int, parameters) -> None:
        """Reconstruct and save model weights received from Flower server."""
        ndarrays = parameters_to_ndarrays(parameters)
        model = nnUNet_CustomNetwork()
        model_state = model.state_dict()

        for idx, key in enumerate(model_state.keys()):
            model_state[key] = torch.tensor(ndarrays[idx])

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        file_name = f"model_state_round_{server_round}_{timestamp}.pth"
        torch.save(model_state, self.save_path / file_name)

    def _store_results(self, tag: str, results_dict: dict) -> None:
        """Store results in memory and write them to a JSON file."""
        self.results.setdefault(tag, []).append(results_dict)

        results_file = self.save_path / "results.json"
        with results_file.open("w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)

    def store_results_and_log(self, server_round: int, tag: str, results_dict: dict) -> None:
        """Store results and optionally log them externally (e.g., W&B)."""
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

    def evaluate(self, server_round: int, parameters):
        """Save the model at each evaluation step. No aggregated result returned."""
        self.save_fl_model(server_round, parameters)
        return None

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and store them."""
        result = super().aggregate_evaluate(server_round, results, failures)
        loss, metrics = result

        self.store_results_and_log(
            server_round=server_round,
            tag="fed_evaluate",
            results_dict={"fl_loss": loss, **metrics},
        )
        return loss, metrics