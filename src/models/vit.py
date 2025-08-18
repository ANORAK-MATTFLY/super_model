from torchvision.models.vision_transformer import VisionTransformer


from torchvision import models
from .interfaces.interface import IModel


class VitModel(IModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._name = config["name"]
        model_params = config["params"]

        self.model: VisionTransformer = models.vit_b_16()

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x):
        return self.model(x)
