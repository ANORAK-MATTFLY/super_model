from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, Generic

T = TypeVar(name="T")
E = TypeVar(name="E")
K = TypeVar(name="K")


class AbstractConvolutionalNeuralNetwork(ABC, Generic[T, E, K]):
    @abstractmethod
    def flatten_layer(self, vector: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def layer_pooling(
        self, convolutional_layers: Sequence[K], base_tensor: T
    ) -> T | None:
        raise NotImplementedError
