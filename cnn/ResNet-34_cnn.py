import tensorflow.keras as keras
import os
from sklearn.metrics import accuracy_score
import numpy as np

# current directory and data file name
model_path = os.path.abspath(f"{os.getcwd()}/data")

# Load fashion mnist dataset
(
    (X_train_full, y_train_full),
    (X_test, y_test),
) = keras.datasets.fashion_mnist.load_data()

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


# A residual Unit class. I is a Layer.
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(
                filters, 3, strides=strides, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(
                    filters, 1, strides=strides, padding="same", use_bias=False
                ),
                keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)


# fashion mnist dataset

# Define Layers
input_layer = keras.layers.Input(shape=[28, 28, 1], name="input")
conv_1 = keras.layers.Conv2D(
    64, 7, strides=2, padding="same", use_bias=False, name="conv_1"
)(input_layer)
bn_1 = keras.layers.BatchNormalization(name="bn_1")(conv_1)
acti_1 = keras.layers.Activation("relu", name="acti_1")(bn_1)
max_pool_2d_1 = keras.layers.MaxPool2D(
    pool_size=3, strides=2, padding="same", name="max_pool_2d_1"
)(acti_1)

prev_filters = 64
prev_layer = max_pool_2d_1
filter_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
for f, filters in enumerate(filter_list):
    strides = 1 if filters == prev_filters else 2
    current_layer = ResidualUnit(filters, strides=strides, name=f"residual_{f}")(
        prev_layer
    )

    prev_layer = current_layer
    prev_filters = filters

global_1 = keras.layers.GlobalAvgPool2D(name="global_avg_pool")(prev_layer)
flatten_1 = keras.layers.Flatten(name="plattfisch")(global_1)
output = keras.layers.Dense(10, activation="softmax", name="output")(flatten_1)

# Build model
model = keras.Model(inputs=[input_layer], outputs=[output])
model.summary()

# Specify optimizer
optimizer = keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, name="Nadam",
)

# Compile model
model.compile(
    loss=["sparse_categorical_crossentropy"], optimizer=optimizer, metrics=["accuracy"]
)

# Define callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(model_path, "best_resnet_model.h5"), save_best_only=True,
)
earlystop_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

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
