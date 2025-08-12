from __future__ import annotations

from typing import Callable, TypeAlias
import numpy as np
from numpy.typing import ArrayLike
from torch import Tensor
from typing import TypeVar

T = TypeVar("T")

TypeIterableArray: TypeAlias = Tensor | ArrayLike


class imp:
    def layer_pool(self):
        print("Hello")


class impTwo:
    def layer_pool(self):
        print("Hello")


Plugin: TypeAlias = imp | impTwo


class ConvolutionalNeuralNetwork:
    def __init__(self, plugin: Plugin) -> None:
        self.plugin = plugin

    def layer_pooling(self) -> TypeIterableArray:
        data: TypeIterableArray = np.array([])
        sh = self.plugin.layer_pool()
        return data

    def flattenLayer(self, layer: TypeIterableArray) -> TypeIterableArray | None:
        pass

    def forward(self, layer: TypeIterableArray) -> TypeIterableArray | None:
        pass

    def train(self, layer: TypeIterableArray) -> TypeIterableArray | None:
        pass
