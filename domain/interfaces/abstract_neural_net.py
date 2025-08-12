from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar(name="T")
E = TypeVar(name="E")


class AbstractNeuralNetwork(ABC, Generic[T, E]):
    @abstractmethod
    def forward(self, layer: T) -> T | None:
        raise NotImplementedError

    @abstractmethod
    def train_model(self, Image_sample: T, y_sample: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, Image_test_sample: T, y_test: T) -> None:
        raise NotImplementedError
