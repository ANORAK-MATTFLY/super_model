from logging import raiseExceptions
from torch._tensor import Tensor
from torch.nn.modules.conv import Conv2d


from collections.abc import Sequence


import torch as tf
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.nn.modules.loss import MSELoss
from torch.optim.adam import Adam

from typing_extensions import override
from tqdm import tqdm
import numpy as np
from .interfaces.abstract_conv_net import AbstractConvolutionalNeuralNetwork
from .interfaces.abstract_neural_net import AbstractNeuralNetwork


from numpy._typing._array_like import NDArray


class NeuralNetWorkWithPytorch(
    # Implements
    AbstractNeuralNetwork[Tensor, NDArray[np.uint8]],
    AbstractConvolutionalNeuralNetwork[Tensor, NDArray[np.uint8], Conv2d],
    # Extends
    nn.Module,
):
    def __init__(self) -> None:
        super().__init__()
        self.convolutional_layer_one: Conv2d = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5
        )
        self.convolutional_layer_two: Conv2d = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5
        )
        self.convolutional_layer_three: Conv2d = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5
        )

        base_tensor: Tensor = tf.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = 0
        self.layer_pooling(base_tensor)
        self.fully_connected_layer_one: nn.Linear = nn.Linear(self._to_linear, 512)
        # Final layer
        self.fully_connected_layer_two = nn.Linear(512, 2)

    @override
    def layer_pooling(self, base_tensor: Tensor) -> Tensor | None:
        output_layer_one: Tensor = F.max_pool2d(
            F.relu(self.convolutional_layer_one(base_tensor)), (2, 2)
        )
        output_layer_two: Tensor = F.max_pool2d(
            F.relu(self.convolutional_layer_two(output_layer_one)), (2, 2)
        )
        layer_pool: Tensor = F.max_pool2d(
            F.relu(self.convolutional_layer_three(output_layer_two)), (2, 2)
        )

        if self._to_linear == 0:
            layer_one_shape = layer_pool[0].shape[0]
            layer_two_shape = layer_pool[0].shape[1]
            layer_three_shape = layer_pool[0].shape[2]
            product_of_layer_dimensions = (
                layer_one_shape * layer_two_shape * layer_three_shape
            )
            self._to_linear = product_of_layer_dimensions
        return layer_pool

    @override
    def flatten_layer(self, vector: Tensor) -> Tensor | None:
        # flattenLayer: Tensor = tf.flatten(vector, start_dim=1)
        # return flattenLayer
        layer_hat = self.layer_pooling(vector)
        if self._to_linear != 0:
            if layer_hat != None:
                layer_hat = layer_hat.view(-1, self._to_linear)
                return layer_hat
        print("Failed to flatten layer")
        return None

    @override
    def forward(self, layer: Tensor) -> Tensor | None:
        flatten_final_layer: Tensor | None = self.flatten_layer(layer)
        if flatten_final_layer is not None:
            flatten_final_layer = F.relu(
                input=self.fully_connected_layer_one(flatten_final_layer)
            )
            final_layer = self.fully_connected_layer_two(flatten_final_layer)
            return F.softmax(final_layer, dim=1)

    @override
    def train_model(self, Image_sample: Tensor, y_sample: Tensor) -> None:
        optimizer: Adam = optim.Adam(self.parameters(), lr=0.001)
        loss_func: MSELoss = nn.MSELoss()

        BATCH_SIZE = 100
        EPOCHS = 9
        loss = 0
        for idx in range(EPOCHS):
            print("Epoch:", idx)
            for index in tqdm(range(0, len(Image_sample), BATCH_SIZE)):
                batch_images = Image_sample[index : index + BATCH_SIZE].view(
                    -1, 1, 50, 50
                )
                batch_y = y_sample[index : index + BATCH_SIZE]
                self.zero_grad()
                outputs = self(batch_images)
                loss = loss_func(outputs, batch_y)
                loss.backward()
                optimizer.step()

    @override
    def evaluate(self, Image_test_sample: Tensor, y_test: Tensor) -> None:
        correct = 0
        total = 0
        with tf.no_grad():
            for index in tqdm(range(len(Image_test_sample))):
                real_class = tf.argmax(y_test[index])
                net_out = self(Image_test_sample[index].view(-1, 1, 50, 50))[0]

                predicted_class: Tensor = tf.argmax(net_out)
                if predicted_class == real_class:
                    correct += 1
                total += 1
        print(f"Accuracy: {(round(correct/total,3)*100)}%")
