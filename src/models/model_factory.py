import yaml
from pathlib import Path
from phys_net import PhysNet

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


MODEL_LIST = {
    "physnet": PhysNet
}


def build_model():
    model_name = cfg["model"]["name"].lower()
    params = cfg["model"].get("params", {})

    if model_name not in MODEL_LIST:
        raise ValueError(f"Model '{model_name}' is not in MODEL_LIST.")

    model_class = MODEL_LIST[model_name]
    model = model_class(**params)

    return model
