from torch.nn.modules.module import Module


from torchvision import models
from .interfaces.interface import IModel


class ResNetModel(IModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._name = config["name"]
        model_params = config["params"]

        self.model: Module = models.get_model(self._name, **model_params)

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x):
        return self.model(x)
