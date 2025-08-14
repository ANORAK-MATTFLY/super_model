import os
import cv2

from torch._tensor import Tensor
from tqdm import tqdm
import numpy as np


REBUILD_DATA = False

BASE_DIR = "./deep_learning/neural_nets"


class DataPrep:
    IMG_SIZE = 50
    CATS = f"./data/PetImages/Cat"
    DOGS = f"./data/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    dog_count = 0
    cat_count = 0

    def build_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(
                            img, (self.IMG_SIZE, self.IMG_SIZE)
                        )  # pyright: ignore[reportCallIssue]
                        self.training_data.append(
                            [np.array(img), np.eye(2)[self.LABELS[label]]]
                        )
                        if label == self.CATS:
                            self.cat_count += 1
                        if label == self.DOGS:
                            self.dog_count += 1
                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save(
            "training_data.npy",
            np.array(self.training_data, dtype=object),
            True,
            fix_imports=True,
        )
        print("CATS: ", self.cat_count)
        print("DOGS: ", self.dog_count + self.cat_count)

    def data_sampling(self, path_to_training_data: str) -> dict[str, Tensor]:

        training_data = np.load(file=path_to_training_data, allow_pickle=True)
        images = Tensor(np.array([image[0] for image in training_data]))
        images = images / 255.0
        y = Tensor(np.array([y[1] for y in training_data]))

        # The percentage of data we take from the set for sampling
        sample_percent = 0.1
        sample_end_index = int(len(images) * sample_percent)
        Image_sample = images[:-sample_end_index]
        y_sample = y[:-sample_end_index]

        Image_test_sample = images[-sample_end_index:]
        y_test = y[-sample_end_index:]
        samples: dict[str, Tensor] = {
            "Image_sample": Image_sample,
            "Image_test_sample": Image_test_sample,
            "y_sample": y_sample,
            "y_test": y_test,
        }

        return samples


if REBUILD_DATA:
    dogs_vs_cats = DataPrep()
    dogs_vs_cats.build_training_data()
