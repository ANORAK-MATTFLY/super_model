from abc import ABC, abstractmethod
import torch.nn as nn


class IModel(nn.Module, ABC):
    """
    The abstract base class for all models in the system.
    It inherits from nn.Module for PyTorch compatibility and defines
    a contract for what our training engine needs.
    """

    @abstractmethod
    def __init__(self, config: dict):
        """Initializes the model using parameters from a config dict."""
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the model."""
        raise NotImplementedError
