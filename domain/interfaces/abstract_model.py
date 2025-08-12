from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar(name="T")


class AbstractModel(ABC, Generic[T]):

    @abstractmethod
    def train_model(self, Image_sample: T, y_sample: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, Image_test_sample: T, y_test: T) -> None:
        raise NotImplementedError
