"""flwr-ntnu-zgt: A Flower / PyTorch app."""
import json
from importlib.resources import files

plan_data = json.loads(
    files("flwr_ntnu_zgt").joinpath("plans.json").read_text()
)