
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# At the beginning of the program
distribution = tf.distribute.MultiWorkerMirroredStrategy()

resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print("Starting task {}{}".format(resolver.task_type, resolver.task_id))

# Only worker #0 will write checkpoints and log to TensorBoard
if resolver.task_id == 0:
    root_logdir = os.path.join(os.curdir, "my_mnist_multiworker_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_dir = os.path.join(root_logdir, run_id)
    callbacks = [
        keras.callbacks.TensorBoard(run_dir),
        keras.callbacks.ModelCheckpoint("my_mnist_multiworker_model.h5",
                                        save_best_only=True),
    ]
else:
    callbacks = []

# Load and prepare the MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis] / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

with distribution.scope():
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                            padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"), 
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=1e-2),
                  metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          epochs=10, callbacks=callbacks)
