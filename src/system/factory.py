from typing import Dict, Type

from src.models.vit import VitModel
from src.models.interfaces.interface import IModel
from src.models.resnet import ResNetModel


# When you add a new model, you just import it here.


MODEL_REGISTRY: Dict[str, Type[IModel]] = {
    "resnet18": ResNetModel,
    "resnet50": ResNetModel,
    "vit_b_16": VitModel,
}


def get_model(config: dict) -> IModel:
    """
    Model Factory.
    Takes a config dictionary, looks up the model name in the registry,
    and returns an instantiated model object.
    """
    model_name = config["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(config)
