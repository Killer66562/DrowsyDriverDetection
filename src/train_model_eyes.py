from keras.api import Model, Sequential
from keras.api.layers import Input, Dropout, Dense, Conv2D, MaxPool2D, Flatten, \
    Resizing, Rescaling, RandomFlip, RandomRotation, RandomBrightness, RandomZoom
from keras.api.optimizers import Adam
from keras.api.utils import image_dataset_from_directory
from keras.api.applications import MobileNetV3Small

from datetime import datetime, timezone, timedelta

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils import show_image_from_dataset


SEED = 42

IMG_ORIGINAL_HEIGHT = 79
IMG_ORIGINAL_WIDTH = 79

IMG_HEIGHT = 79
IMG_WIDTH = 79

BATCH_SIZE = 250
EPOCHS = 15
LEARNING_RATE = 0.001

DATASETS_PATH = "datasets"

DATASETS_TRAIN_PATH = os.path.join(DATASETS_PATH, "eyes", "train")
DATASETS_TEST_PATH = os.path.join(DATASETS_PATH, "eyes", "test")

MODEL_SAVE_PATH = os.path.join("detection_models", "eyes")

COLOR = "grayscale"
CHANNELS = 4 if COLOR == "rgb" else 1 if COLOR == "grayscale" else 3

model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype="float32"), 

    Conv2D(4, (3, 3), activation="relu", padding="same"), 
    Conv2D(4, (3, 3), activation="relu", padding="same"), 
    Conv2D(4, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(8, (3, 3), activation="relu", padding="same"), 
    Conv2D(8, (3, 3), activation="relu", padding="same"), 
    Conv2D(8, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(16, (3, 3), activation="relu", padding="same"), 
    Conv2D(16, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(32, (3, 3), activation="relu", padding="same"), 
    Conv2D(32, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Dropout(0.1), 
    Flatten(), 

    Dense(64, activation='relu'), 
    Dense(16, activation='relu'), 
    Dense(4, activation="relu"), 
    Dense(1, activation="sigmoid")
])

normalization = Sequential([
    Resizing(IMG_HEIGHT, IMG_WIDTH), 
    Rescaling(1. / 255)
])

data_augmentation = Sequential([
    RandomZoom(0.5), 
    RandomFlip("horizontal"), 
    RandomRotation(0.2), 
    RandomBrightness(0.2)
])

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

dataset_train, dataset_validation = image_dataset_from_directory(
    DATASETS_TRAIN_PATH, 
    color_mode=COLOR, 
    label_mode="binary", 
    image_size=(IMG_ORIGINAL_HEIGHT, IMG_ORIGINAL_WIDTH), 
    class_names=["close eyes", "open eyes"], 
    interpolation="bilinear", 
    subset="both", 
    validation_split=0.2, 
    batch_size=BATCH_SIZE, 
    seed=SEED
)

dataset_test = image_dataset_from_directory(
    DATASETS_TEST_PATH, 
    color_mode=COLOR, 
    label_mode="binary", 
    image_size=(IMG_ORIGINAL_HEIGHT, IMG_ORIGINAL_WIDTH), 
    class_names=["close eyes", "open eyes"], 
    interpolation="bilinear", 
    batch_size=BATCH_SIZE, 
    seed=SEED
)

dataset_train = dataset_train.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
dataset_train = dataset_train.map(lambda x, y: (normalization(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

show_image_from_dataset(dataset_train)

dataset_validation = dataset_validation.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

dataset_test = dataset_test.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

hist = model.fit(
    dataset_train, 
    epochs=EPOCHS, 
    validation_data=dataset_validation, 
)

result = model.evaluate(dataset_test)
print("accuracy=%.4f" % result[1])

current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
model_name = f'model-{current_time}.keras'
model_path = os.path.join(MODEL_SAVE_PATH, model_name)
model.save(model_path)

