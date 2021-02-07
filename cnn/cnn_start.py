
# Import stuff
from sklearn.datasets import load_sample_image
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Set colormap
cmap = "hot"

# Load sample images
china = load_sample_image("china.jpg") / 255 # dims: height x width x channels
flower = load_sample_image("flower.jpg") / 255 # dims: height x width x channels
images = np.array([china, flower]) # dims: n_images (batch size) x height x width x channels
batch_size, height, width, n_image_channels = images.shape

# Plot the first images first channel
plt.imshow(images[0, :, :, 0], cmap=cmap)
plt.show()

# Create filters
filters = np.zeros(shape=(7, 7, n_image_channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # Vertical line
filters[3, :, :, 1] = 1 # Horizontal line

# Apply filters
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

# Plot the convoluted image
plt.imshow(outputs[0, :, :, 0], cmap=cmap) # Vertical filter channel
plt.show()
plt.imshow(outputs[0, :, :, 1], cmap=cmap) # Horizontal filter channel
plt.show()

# Load fashion mnist dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0
class_names = np.array(
    [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
)

# current directory and data file name
model_path = os.path.abspath(f"{os.getcwd()}/data")


# Specify a cnn architecture using the functional API
input_layer =  tf.keras.layers.Input(shape=[28, 28, 1], name="input")
conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding="same", activation="relu", name="conv_1")(input_layer)
pool_1 = tf.keras.layers.MaxPooling2D(2, name="pool_1")(conv_1)
conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_2")(pool_1)
conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_3")(conv_2)
pool_2 = tf.keras.layers.MaxPooling2D(2, name="pool_2")(conv_3)
conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_4")(pool_2)
conv_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_5")(conv_4)
pool_3 = tf.keras.layers.MaxPooling2D(2, name="pool_3")(conv_5)
flatten_1 = tf.keras.layers.Flatten(name="flatten")(pool_3)
dense_1 = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(flatten_1)
dropout_1 = tf.keras.layers.Dropout(0.5, name="dropout_1")(dense_1)
dense_2 = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(dropout_1)
dropout_2 = tf.keras.layers.Dropout(0.5, name="dropout_2")(dense_2)
output = tf.keras.layers.Dense(10, activation="softmax")(dropout_2)

# Build model
model = tf.keras.Model(inputs=[input_layer], outputs=[output])
model.summary()

# Specify optimizer
optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, name="Nadam",
)

# Compile model
model.compile(
    loss=["sparse_categorical_crossentropy"], optimizer=optimizer, metrics=["accuracy"]
)

# Define callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(model_path, "best_model.h5"), save_best_only=True,
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

# Fit the model
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, earlystop_cb],
    batch_size=32,
)

# Predict
y_predicted = model.predict(X_test)
predictions = y_predicted.argmax(axis=1).astype(np.int)

# Get model accuracy
model_accuracy = accuracy_score(y_test, predictions, normalize=True)

loaded_layer = model.get_layer('conv_1')
loaded_layer_output = loaded_layer.outputs
