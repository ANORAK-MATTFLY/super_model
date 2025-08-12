from typing_extensions import override
import torch.nn as nn
from torch._tensor import Tensor
import torch as tf

from torch.nn.modules.conv import Conv2d

from typing import TypeVar

from domain.data_prep import DataPrep
from domain.interfaces.abstract_model import AbstractModel
from domain.torch_conv_net_service import (
    NeuralNetWorkWithPytorch,
)

from dependency_injection import deps

T = TypeVar(name="T")


class IMAGE_CLASSIFICATION_MODEL(AbstractModel):
    def __init__(self) -> None:
        # Initializing a final layer for distribution
        """
        The first argument -1 means 'for any size of tensor'.
        The second (hint 1) represent the entire batch of data.
        The 50 by 50 represents the size of images within the batch of data.
        """
        self.conv: NeuralNetWorkWithPytorch = deps["conv_nn"]

    @override
    def train_model(self, Image_sample: Tensor, y_sample: Tensor):
        self.conv.train_model(Image_sample, y_sample)

    @override
    def evaluate(self, Image_test_sample: Tensor, y_test: Tensor):
        self.conv.evaluate(Image_test_sample, y_test)


def main():
    training_data = DataPrep().data_sampling("./training_data.npy")
    Image_sample = training_data["Image_sample"]
    Image_test_sample = training_data["Image_test_sample"]
    y_sample = training_data["y_sample"]
    y_test = training_data["y_test"]

    IMAGE_CLASSIFICATION_MODEL().train_model(Image_sample, y_sample)
    IMAGE_CLASSIFICATION_MODEL().evaluate(Image_test_sample, y_test)
    pass


main()
