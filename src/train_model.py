from keras.api import Model, Sequential
from keras.api.layers import Input, Dropout, Dense, Conv2D, MaxPool2D, Flatten, \
    Resizing, Rescaling, RandomFlip, RandomRotation, RandomBrightness, RandomZoom
from keras.api.optimizers import Adam
from keras.api.utils import image_dataset_from_directory
from keras.api.applications import ResNet50V2

from datetime import datetime, timezone, timedelta

import os

SEED = 42

IMG_ORIGINAL_HEIGHT = 227
IMG_ORIGINAL_WIDTH = 227

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 200
EPOCHS = 20
LEARNING_RATE = 0.001

DATASETS_PATH = "datasets"


model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 

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

    Conv2D(64, (3, 3), activation="relu", padding="same"), 
    Conv2D(64, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Flatten(), 
    Dropout(0.1), 

    Dense(256, activation='relu'), 
    Dense(16, activation='relu'), 
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

datasets_path = os.path.abspath(DATASETS_PATH)
dataset_train, dataset_validation = image_dataset_from_directory(
    datasets_path, 
    color_mode="rgb", 
    label_mode="binary", 
    image_size=(IMG_ORIGINAL_HEIGHT, IMG_ORIGINAL_WIDTH), 
    class_names=["Non Drowsy", "Drowsy"], 
    interpolation="nearest", 
    batch_size=BATCH_SIZE, 
    validation_split=0.3, 
    subset="both", 
    seed=SEED
)

dataset_train = dataset_train.map(lambda x, y: (data_augmentation(x), y))

dataset_train = dataset_train.map(lambda x, y: (normalization(x), y))
dataset_validation = dataset_validation.map(lambda x, y: (normalization(x), y))

hist = model.fit(
    dataset_train, 
    epochs=EPOCHS, 
    validation_data=dataset_validation, 
)

result = model.evaluate(dataset_validation)
print("accuracy=%.4f" % result[1])

current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
model_name = f'model-{current_time}.keras'
model_path = os.path.join("models", model_name)
model.save(model_path)
