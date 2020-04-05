#!/usr/bin/env python
import joblib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift as imageshift


def shift_image(image_data, direction, distance):
    image_2d = image_data.reshape(28, 28)
    distances = {
        "west": -distance,
        "east": distance,
        "north": -distance,
        "south": distance,
    }
    dimensions = {"west": 1, "east": 1, "north": 0, "south": 0}

    shiftvec = [0, 0]
    shiftvec[dimensions[direction]] = distances[direction]

    # print(
    #     f"Shifting {distance}px to {direction} with Vector ({shiftvec[0]},{shiftvec[1]})"
    # )

    shifted = imageshift(image_2d, shiftvec)
    return shifted.reshape(image_data.shape)


# def display_image(image_data):
#     image_2d = image_data.reshape(28, 28)
#     plt.imshow(image_2d, cmap=matplotlib.cm.binary, interpolation="nearest")
#     plt.axis("off")
#     plt.show()


training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

# jedes bild
# erzeugen 4 kopien
# pushen wir in ein array
# concat neues array hinten an die trainingsdaten
augmentation_data = np.zeros([training_data.shape[0] * 4, training_data.shape[1]])
for i, image in enumerate(training_data):
    for j, direction in enumerate(["north", "south", "east", "west"]):
        augmentation_data[i * 4 + j, :] = shift_image(image, direction, 10)

# for i in range(4):
#     display_image(augmentation_data[i, :])

augmented_data = np.r_[training_data, augmentation_data]
