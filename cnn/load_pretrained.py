import tensorflow.keras as keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input


def load_images_for_keras(filenames, img_path, target_size=(224, 224)):
    """ load all images from imgPath for ResNet 50 and return ready for prediction"""
    images = []

    for filename in filenames:

        img = keras.preprocessing.image.load_img(
            os.path.join(img_path, filename), target_size=target_size
        )
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = keras.applications.resnet50.preprocess_input(img)

        images.append(img)

    return images


# Load example images
china = load_sample_image("china.jpg") / 255  # dims: height x width x channels
flower = load_sample_image("flower.jpg") / 255  # dims: height x width x channels
example_images = [china, flower]

# Own images path
image_path = os.path.abspath(f"{os.getcwd()}/images")
image_filenames = sorted(os.listdir(image_path))

# Load own images
loaded_images = load_images_for_keras(image_filenames, image_path)

# Load ResNet 50 pretrained on imagenet dataset
model = keras.applications.resnet50.ResNet50(weights="imagenet")

# Show predicted classes
for i, image in enumerate(loaded_images):

    # Predict
    Y_pred = model.predict(image)

    # decode and display
    decoded_predictions = keras.applications.resnet50.decode_predictions(Y_pred, top=5)

    print(f"Prediction for {image_filenames[i]}: ")
    for class_id, name, prediction in decoded_predictions[0]:
        print(f"{class_id:^30}|{name:^30}|{prediction*100}%")
